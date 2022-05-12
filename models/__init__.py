# ------------------------------------------------------------------------
# Copied from Conditional DETR: https://github.com/Atten4Vis/ConditionalDETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------

from .gestalt_detr import build


def build_model(args):
    return build(args)
