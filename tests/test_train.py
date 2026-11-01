from train import create_base_model


def test_backbone_is_tri_axial_by_default():
    assert create_base_model(450, 115).channels == 3


def test_backbone_takes_the_channel_count_it_is_given():
    assert create_base_model(1800, 115, channels=6).channels == 6
