##############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
##############################################################################

"""A large_image tile source backed by ZEISS libCZI through pylibCZIrw."""

import contextlib
import importlib.metadata
import math
import os
import threading

import numpy as np
import PIL.Image

from large_image.cache_util import LruCacheMetaclass, methodcache
from large_image.constants import TILE_FORMAT_IMAGE, TILE_FORMAT_NUMPY, SourcePriority
from large_image.exceptions import TileSourceError, TileSourceFileNotFoundError
from large_image.tilesource import FileTileSource, utilities
from large_image.tilesource.tileiterator import TileIterator

czi = None
_requiredPerformanceApiVersion = 1

with contextlib.suppress(importlib.metadata.PackageNotFoundError):
    __version__ = importlib.metadata.version(__name__)


def _lazyImport():
    """Import pylibCZIrw only when a CZI source is opened."""
    global czi

    if czi is None:
        try:
            from pylibCZIrw import czi as czi_module
        except ImportError:
            msg = 'pylibCZIrw module not found.'
            raise TileSourceError(msg) from None
        czi = czi_module


def _requirePerformanceAPI():
    """Ensure the native extension contains the optimized libCZI contract."""
    version = int(getattr(czi, 'PERFORMANCE_API_VERSION', 0))
    if version < _requiredPerformanceApiVersion:
        msg = (
            'The CZI source requires pylibCZIrw built with the optimized '
            f'libCZI API version {_requiredPerformanceApiVersion} or newer; '
            f'the loaded extension reports version {version}.')
        raise TileSourceError(msg)
    return version


def _asList(value):
    """Return an XML metadata value as a list."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _findValues(value, key):
    """Yield values of case-insensitive keys in nested XML metadata."""
    if isinstance(value, dict):
        for entryKey, entryValue in value.items():
            if entryKey.lstrip('@').lower() == key.lower():
                yield entryValue
            yield from _findValues(entryValue, key)
    elif isinstance(value, list):
        for entry in value:
            yield from _findValues(entry, key)


def _rectToTuple(rect):
    """Convert a pylibCZIrw Rectangle or tuple to a plain tuple."""
    if all(hasattr(rect, key) for key in ('x', 'y', 'w', 'h')):
        return (int(rect.x), int(rect.y), int(rect.w), int(rect.h))
    return tuple(int(value) for value in rect)


class CZIFileTileSource(FileTileSource, metaclass=LruCacheMetaclass):
    """Provide tiled, multiscale access to Carl Zeiss Image (CZI) files."""

    cacheName = 'tilesource'
    name = 'czi'
    extensions = {
        None: SourcePriority.FALLBACK,
        'czi': SourcePriority.PREFERRED,
    }
    mimeTypes = {
        None: SourcePriority.FALLBACK,
        'application/x-czi': SourcePriority.PREFERRED,
        'image/czi': SourcePriority.PREFERRED,
        'image/x-czi': SourcePriority.PREFERRED,
    }

    _tileSize = 512
    _axisOrder = ('C', 'Z', 'T', 'R', 'I', 'H', 'V', 'B')
    _dtypeForPixelType = {
        'bgr24': np.uint8,
        'bgr48': np.uint16,
        'bgr96float': np.float32,
        'gray8': np.uint8,
        'gray16': np.uint16,
        'gray32float': np.float32,
    }

    def __init__(self, path, **kwargs):
        """Initialize a CZI tile source from a filesystem path."""
        cacheBytes = max(0, int(kwargs.pop('cziCacheBytes', 512 * 1024 ** 2)))
        readThreads = max(1, int(kwargs.pop('cziReadThreads', 2)))
        useNativeTileSize = bool(kwargs.pop('cziNativeTileSize', True))
        self._directRegionMaxPixels = max(
            0, int(kwargs.pop('cziDirectRegionMaxPixels', 64 * 1024 ** 2)))
        super().__init__(path, **kwargs)

        self._largeImagePath = str(self._getLargeImagePath())
        if not os.path.isfile(self._largeImagePath):
            raise TileSourceFileNotFoundError(self._largeImagePath)

        _lazyImport()
        self._cziPerformanceApiVersion = _requirePerformanceAPI()
        self._cziContext = None
        try:
            self._openCZI(cacheBytes, readThreads)
            rect = self._readInitialCZIState()
        except Exception as exc:
            self._close()
            self.logger.debug('File cannot be opened via pylibCZIrw: %r', exc)
            msg = 'File cannot be opened via the CZI source.'
            raise TileSourceError(msg) from exc

        self._originX, self._originY, self.sizeX, self.sizeY = rect
        if self.sizeX <= 0 or self.sizeY <= 0:
            self._close()
            msg = 'CZI image dimensions must be positive.'
            raise TileSourceError(msg)

        self.tileWidth = self.tileHeight = self._tileSize
        self._setNativeTileSize(useNativeTileSize)
        self.levels = max(1, int(math.ceil(math.log2(max(
            self.sizeX / self.tileWidth, self.sizeY / self.tileHeight)))) + 1)
        self._sceneIds = sorted(self._sceneRectangles) or [None]
        self._axisCounts = {
            axis: max(1, int(self._dimensionBounds.get(axis, (0, 1))[1] -
                             self._dimensionBounds.get(axis, (0, 1))[0]))
            for axis in self._axisOrder
        }
        self._axisBases = {}
        basis = 1
        for axis in self._axisOrder:
            self._axisBases[axis] = basis
            basis *= self._axisCounts[axis]
        self._sceneBasis = basis
        self._frameCount = basis * len(self._sceneIds)
        self._channels = None
        self._magnification = None
        self._metadataLock = threading.RLock()
        self._setPixelCharacteristics()

    def _openCZI(self, cacheBytes, readThreads):
        """Open pylibCZIrw with bounded caching and decode concurrency."""
        cacheOptions = czi.CacheOptions()
        cacheOptions.max_memory_usage = cacheBytes
        readerOptions = czi.ReaderOptions()
        readerOptions.max_concurrent_subblock_reads = readThreads
        self._cziContext = czi.open_czi(
            self._largeImagePath,
            cache_options=cacheOptions,
            reader_options=readerOptions,
        )
        self._czi = self._cziContext.__enter__()

    def _readInitialCZIState(self):
        """Read the compact native state needed to expose tiles and frames."""
        self._dimensionBounds = dict(self._czi.total_bounding_box_no_pyramid)
        rect = _rectToTuple(self._czi.total_bounding_rectangle_no_pyramid)
        self._sceneRectangles = {}
        self._pixelTypes = {}
        self._cziMetadata = None
        try:
            self._sceneRectangles = {
                int(key): _rectToTuple(value)
                for key, value in self._czi.scenes_bounding_rectangle_no_pyramid.items()
            }
        except Exception as exc:
            self.logger.debug('Cannot read CZI scene bounds: %r', exc)
        try:
            self._pixelTypes = dict(self._czi.pixel_types)
        except Exception as exc:
            self.logger.debug('Cannot read CZI pixel types: %r', exc)
        return rect

    def _setNativeTileSize(self, enabled):
        """Use the common layer-zero subblock dimensions when available."""
        if not enabled or not hasattr(self._czi, 'recommended_tile_size'):
            return
        try:
            nativeWidth, nativeHeight = self._czi.recommended_tile_size
            if 64 <= nativeWidth <= 4096 and 64 <= nativeHeight <= 4096:
                self.tileWidth, self.tileHeight = int(nativeWidth), int(nativeHeight)
        except Exception as exc:
            self.logger.debug('Cannot determine native CZI tile size: %r', exc)

    def _close(self):
        """Close the pylibCZIrw context, if it was opened."""
        context = getattr(self, '_cziContext', None)
        self._cziContext = None
        if context is not None:
            with contextlib.suppress(Exception):
                context.__exit__(None, None, None)

    def __del__(self):
        self._close()

    def _getChannelNames(self):
        """Extract channel names from CZI XML metadata."""
        self._loadMetadata()
        try:
            metadata = self._cziMetadata['ImageDocument']['Metadata']
            channels = metadata['Information']['Image']['Dimensions']['Channels']['Channel']
            result = []
            for idx, channel in enumerate(_asList(channels)):
                if not isinstance(channel, dict):
                    result.append(str(channel))
                    continue
                result.append(str(
                    channel.get('@Name') or channel.get('Name') or
                    channel.get('@Id') or channel.get('Id') or f'Channel {idx}'))
            if len(result) >= self._axisCounts['C']:
                return result
        except (KeyError, TypeError):
            pass
        return None

    def _getMagnification(self):
        """Extract physical pixel size and objective magnification."""
        self._loadMetadata()
        result = {'magnification': None, 'mm_x': None, 'mm_y': None}
        try:
            metadata = self._cziMetadata['ImageDocument']['Metadata']
            scaling = metadata.get('Scaling', {})
            for distance in _findValues(scaling, 'Distance'):
                for entry in _asList(distance):
                    if not isinstance(entry, dict):
                        continue
                    axis = str(entry.get('@Id', entry.get('Id', ''))).lower()
                    value = entry.get('Value')
                    if isinstance(value, dict):
                        value = value.get('#text')
                    if axis in {'x', 'y'} and value is not None:
                        # CZI scaling distances are expressed in meters.
                        result['mm_' + axis] = float(value) * 1000
            for value in _findValues(metadata.get('Information', {}), 'NominalMagnification'):
                if isinstance(value, dict):
                    value = value.get('#text')
                if value is not None:
                    result['magnification'] = float(value)
                    break
        except (KeyError, TypeError, ValueError):
            pass
        if result['magnification'] is None and result['mm_x']:
            result['magnification'] = 0.01 / result['mm_x']
        if result['mm_y'] is None:
            result['mm_y'] = result['mm_x']
        return result

    def _loadMetadata(self):
        """Load and parse XML metadata on first use."""
        if self._cziMetadata is not None:
            return
        with self._metadataLock:
            if self._cziMetadata is None:
                try:
                    self._cziMetadata = self._czi.metadata
                except Exception as exc:
                    self.logger.debug('Cannot parse CZI XML metadata: %r', exc)
                    self._cziMetadata = {}

    def _setPixelCharacteristics(self):
        """Set dtype and band count without forcing an image read."""
        pixelTypes = [str(value).lower() for value in self._pixelTypes.values()]
        dtypes = [self._dtypeForPixelType[value]
                  for value in pixelTypes if value in self._dtypeForPixelType]
        if dtypes:
            self._dtype = np.result_type(*dtypes)
        if pixelTypes:
            self._bandCount = 3 if any(value.startswith('bgr') for value in pixelTypes) else 1

    def _getOutputPixelType(self, pixelType):
        """Select a common bit depth while retaining gray or BGR layout."""
        if not pixelType or self._dtype is None:
            return None
        isBgr = pixelType.startswith('bgr')
        dtype = np.dtype(self._dtype)
        if dtype == np.dtype(np.uint8):
            return 'Bgr24' if isBgr else 'Gray8'
        if dtype == np.dtype(np.uint16):
            return 'Bgr48' if isBgr else 'Gray16'
        if dtype == np.dtype(np.float32):
            return 'Bgr96Float' if isBgr else 'Gray32Float'
        return None

    def getNativeMagnification(self):
        """Return objective magnification and physical pixel dimensions."""
        if self._magnification is None:
            self._magnification = self._getMagnification()
        return self._magnification.copy()

    def _frameValues(self, frame):
        """Convert a large_image frame number to CZI plane and scene values."""
        sceneOffset, planeOffset = divmod(frame, self._sceneBasis)
        plane = {}
        indices = {}
        for axis in self._axisOrder:
            index = (planeOffset // self._axisBases[axis]) % self._axisCounts[axis]
            indices[axis] = index
            if self._axisCounts[axis] > 1:
                plane[axis] = self._dimensionBounds.get(axis, (0, 1))[0] + index
        return plane, self._sceneIds[sceneOffset], indices, sceneOffset

    def getMetadata(self):
        """Return standard large_image metadata, including CZI dimensions."""
        if not hasattr(self, '_computedMetadata'):
            if self._channels is None:
                self._channels = self._getChannelNames()
            result = super().getMetadata()
            if self._frameCount > 1:
                result['frames'] = frames = []
                for frame in range(self._frameCount):
                    _plane, _scene, indices, sceneOffset = self._frameValues(frame)
                    entry = {
                        'Index' + axis: index
                        for axis, index in indices.items()
                        if self._axisCounts[axis] > 1
                    }
                    if len(self._sceneIds) > 1:
                        entry['IndexXY'] = sceneOffset
                    frames.append(entry)
                self._addMetadataFrameInformation(result, self._channels)
            self._computedMetadata = result
        return self._computedMetadata

    def getInternalMetadata(self, **kwargs):
        """Return CZI metadata, dimensions, scene bounds, and pixel types."""
        self._loadMetadata()
        return {
            'czi': self._cziMetadata,
            'czi_dimensions': self._dimensionBounds,
            'czi_scenes': self._sceneRectangles,
            'czi_pixel_types': self._pixelTypes,
            'czi_performance_api_version': self._cziPerformanceApiVersion,
        }

    def _readCZI(self, x0, y0, x1, y1, zoom, frame):
        """Read and normalize a CZI rectangle without serializing callers."""
        plane, scene, _indices, _sceneOffset = self._frameValues(frame)
        channel = plane.get('C', self._dimensionBounds.get('C', (0, 1))[0])
        pixelType = str(self._pixelTypes.get(channel, '')).lower()
        tile = self._czi.read(
            roi=(self._originX + x0, self._originY + y0, x1 - x0, y1 - y0),
            plane=plane or None,
            scene=scene,
            zoom=zoom,
            pixel_type=self._getOutputPixelType(pixelType),
        )
        tile = np.asarray(tile)
        if tile.ndim == 2:
            tile = tile[:, :, np.newaxis]
        if pixelType.startswith('bgr') and tile.ndim == 3 and tile.shape[2] == 3:
            tile = tile[:, :, ::-1]
        return tile

    @methodcache()
    def getTile(self, x, y, z, pilImageAllowed=False, numpyAllowed=False, **kwargs):
        """Read one large_image tile through libCZI's scaling accessor."""
        frame = self._getFrame(**kwargs)
        self._xyzInRange(x, y, z, frame, self._frameCount)
        x0, y0, x1, y1, step = self._xyzToCorners(x, y, z)
        try:
            tile = self._readCZI(x0, y0, x1, y1, 1.0 / step, frame)
        except Exception as exc:
            self.logger.debug('Failed to read CZI tile: %r', exc)
            msg = 'Failed to read CZI tile.'
            raise TileSourceError(msg) from exc
        return self._outputTile(
            tile, TILE_FORMAT_NUMPY, x, y, z, pilImageAllowed, numpyAllowed, **kwargs)

    def getRegion(self, format=(TILE_FORMAT_IMAGE,), **kwargs):
        """Read ordinary regions with a single native composite operation."""
        if not isinstance(format, (tuple, set, list)):
            format = (format,)
        unsupported = any(key in kwargs for key in (
            'tile_offset', 'tile_overlap', 'tile_position', 'tile_size'))
        if (unsupported or kwargs.get('fill') or
                (TILE_FORMAT_IMAGE in format and kwargs.get('encoding') == 'TILED') or
                getattr(self, '_style', None) is not None):
            return super().getRegion(format=format, **kwargs)

        iteratorKwargs = kwargs.copy()
        resample = iteratorKwargs.pop('resample', True)
        iterator = TileIterator(
            self, format=TILE_FORMAT_NUMPY, resample=None, **iteratorKwargs)
        if iterator.info is None:
            return super().getRegion(format=format, **kwargs)
        info = iterator.info
        outWidth = int(math.floor(info['output']['width']))
        outHeight = int(math.floor(info['output']['height']))
        regionWidth = int(info['region']['width'])
        regionHeight = int(info['region']['height'])
        if (outWidth <= 0 or outHeight <= 0 or
                max(outWidth * outHeight, regionWidth * regionHeight) >
                self._directRegionMaxPixels):
            return super().getRegion(format=format, **kwargs)

        factor = 2 ** (self.levels - 1 - info['level'])
        x0 = int(info['region']['left'] * factor)
        y0 = int(info['region']['top'] * factor)
        x1 = int(info['region']['right'] * factor)
        y1 = int(info['region']['bottom'] * factor)
        zoom = 1.0 / factor
        try:
            image = self._readCZI(
                x0, y0, x1, y1, zoom, self._getFrame(frame=info['frame']))
        except Exception as exc:
            self.logger.debug('Direct CZI region read failed; using tiles: %r', exc)
            return super().getRegion(format=format, **kwargs)

        if image.shape[1] != outWidth or image.shape[0] != outHeight:
            dtype = image.dtype
            if dtype == np.uint8 or (resample is not None and (
                    dtype != np.uint16 or image.shape[-1] != 1)):
                image = utilities._imageToPIL(image, info['mode']).resize(
                    (outWidth, outHeight),
                    getattr(PIL.Image, 'Resampling', PIL.Image).NEAREST
                    if resample is None else
                    getattr(PIL.Image, 'Resampling', PIL.Image).BICUBIC
                    if outWidth > image.shape[1] else
                    getattr(PIL.Image, 'Resampling', PIL.Image).LANCZOS)
                if dtype == np.uint16 and TILE_FORMAT_NUMPY in format:
                    image = utilities._imageToNumpy(image)[0].astype(dtype) * 257
            else:
                cols = [int(idx * image.shape[1] / outWidth) for idx in range(outWidth)]
                rows = [int(idx * image.shape[0] / outHeight) for idx in range(outHeight)]
                image = np.take(np.take(image, rows, axis=0), cols, axis=1)
        return utilities._encodeImage(image, format=format, **kwargs)


def open(*args, **kwargs):
    """Create a CZI tile source instance."""
    return CZIFileTileSource(*args, **kwargs)


def canRead(*args, **kwargs):
    """Return whether an input can be read as CZI."""
    return CZIFileTileSource.canRead(*args, **kwargs)
