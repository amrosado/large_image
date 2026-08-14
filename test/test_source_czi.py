import contextlib
import types
from typing import NamedTuple

import large_image_source_czi
import numpy as np
import pytest

import large_image
from large_image.exceptions import TileSourceFileNotFoundError

from . import utilities
from .datastore import datastore


class Rectangle(NamedTuple):
    x: int
    y: int
    w: int
    h: int


class FakeCZIReader:
    total_bounding_box_no_pyramid = {
        'X': (-100, 924),
        'Y': (50, 650),
        'C': (0, 2),
        'Z': (0, 3),
        'T': (0, 1),
    }
    total_bounding_rectangle_no_pyramid = Rectangle(-100, 50, 1024, 600)
    scenes_bounding_rectangle_no_pyramid = {
        0: Rectangle(-100, 50, 400, 600),
        1: Rectangle(300, 50, 624, 600),
    }
    pixel_types = {0: 'Gray8', 1: 'Bgr48'}
    metadata = {
        'ImageDocument': {
            'Metadata': {
                'Information': {
                    'Image': {
                        'Dimensions': {
                            'Channels': {
                                'Channel': [
                                    {'@Id': 'Channel:0', '@Name': 'DAPI'},
                                    {'@Id': 'Channel:1', '@Name': 'Brightfield'},
                                ],
                            },
                        },
                    },
                    'Instrument': {
                        'Objectives': {
                            'Objective': {'NominalMagnification': '40'},
                        },
                    },
                },
                'Scaling': {
                    'Items': {
                        'Distance': [
                            {'@Id': 'X', 'Value': '2.5e-7'},
                            {'@Id': 'Y', 'Value': {'#text': '5e-7'}},
                        ],
                    },
                },
            },
        },
    }

    def __init__(self):
        self.reads = []

    def read(self, roi=None, plane=None, scene=None, zoom=None, pixel_type=None):
        self.reads.append({
            'roi': roi,
            'plane': plane,
            'scene': scene,
            'zoom': zoom,
            'pixel_type': pixel_type,
        })
        height = int(round(roi[3] * zoom))
        width = int(round(roi[2] * zoom))
        if plane and plane.get('C') == 1:
            result = np.empty((height, width, 3), dtype=np.uint16)
            result[:] = (10, 20, 30)
            return result
        return np.full((height, width, 1), 42, dtype=np.uint16)


@contextlib.contextmanager
def fakeOpenCZI(_path):
    yield FakeCZIReader()


@pytest.fixture
def fakeCZI(monkeypatch):
    module = types.SimpleNamespace(open_czi=fakeOpenCZI)
    monkeypatch.setattr(large_image_source_czi, 'czi', module)
    return module


def testCZIMetadataAndTiles(tmp_path, fakeCZI):
    path = tmp_path / 'sample.czi'
    path.write_bytes(b'not needed by the fake reader')
    source = large_image_source_czi.open(path)

    metadata = source.getMetadata()
    assert metadata['sizeX'] == 1024
    assert metadata['sizeY'] == 600
    assert metadata['tileWidth'] == metadata['tileHeight'] == 512
    assert metadata['levels'] == 2
    assert metadata['dtype'] == 'uint16'
    assert metadata['bandCount'] == 3
    assert metadata['magnification'] == 40
    assert metadata['mm_x'] == pytest.approx(0.00025)
    assert metadata['mm_y'] == pytest.approx(0.0005)
    assert len(metadata['frames']) == 12
    assert metadata['channels'] == ['DAPI', 'Brightfield']
    assert metadata['IndexRange'] == {'IndexC': 2, 'IndexXY': 2, 'IndexZ': 3}
    assert metadata['IndexStride'] == {'IndexC': 1, 'IndexXY': 6, 'IndexZ': 2}
    assert metadata['frames'][7]['IndexC'] == 1
    assert metadata['frames'][7]['IndexXY'] == 1
    assert metadata['frames'][7]['IndexZ'] == 0

    tile = source.getTile(0, 0, 0, frame=7, numpyAllowed=True)
    assert tile.shape == (512, 512, 3)
    assert tuple(tile[0, 0]) == (30, 20, 10)
    assert source._czi.reads[-1] == {
        'roi': (-100, 50, 1024, 600),
        'plane': {'C': 1, 'Z': 0},
        'scene': 1,
        'zoom': 0.5,
        'pixel_type': 'Bgr48',
    }

    source.getTile(0, 0, 1, frame=0, numpyAllowed=True)
    assert source._czi.reads[-1]['pixel_type'] == 'Gray16'

    internal = source.getInternalMetadata()
    assert internal['czi_pixel_types'] == {0: 'Gray8', 1: 'Bgr48'}
    assert internal['czi_scenes'][1] == (300, 50, 624, 600)
    source._close()


def testCZIMissingFile(fakeCZI, tmp_path):
    with pytest.raises(TileSourceFileNotFoundError):
        large_image_source_czi.open(tmp_path / 'missing.czi')


def testTilesFromCZI():
    imagePath = datastore.fetch('HENormalN801.czi')
    source = large_image.open(imagePath)
    assert isinstance(source, large_image_source_czi.CZIFileTileSource)
    metadata = source.getMetadata()

    assert metadata['tileWidth'] == 512
    assert metadata['tileHeight'] == 512
    assert metadata['sizeX'] == 50577
    assert metadata['sizeY'] == 17417
    assert metadata['levels'] == 8
    assert metadata['magnification'] == pytest.approx(20)
    assert metadata['dtype'] == 'uint8'
    assert metadata['bandCount'] == 3
    utilities.checkTilesZXY(source, metadata)
    assert 'czi' in source.getInternalMetadata()
