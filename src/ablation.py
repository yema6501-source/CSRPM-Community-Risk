from __future__ import annotations

ABLATIONS = {
    "full_csrpm": {
        "tensor_factorization": True,
        "spatial_regularization": True,
        "adaptive_dependency": True,
        "temporal_encoder": True,
        "uncertainty_calibration": True,
    },
    "no_tensor_factorization": {
        "tensor_factorization": False,
        "spatial_regularization": True,
        "adaptive_dependency": True,
        "temporal_encoder": True,
        "uncertainty_calibration": True,
    },
    "no_spatial_regularization": {
        "tensor_factorization": True,
        "spatial_regularization": False,
        "adaptive_dependency": True,
        "temporal_encoder": True,
        "uncertainty_calibration": True,
    },
    "static_dependency": {
        "tensor_factorization": True,
        "spatial_regularization": True,
        "adaptive_dependency": False,
        "temporal_encoder": True,
        "uncertainty_calibration": True,
    },
    "no_temporal_encoder": {
        "tensor_factorization": True,
        "spatial_regularization": True,
        "adaptive_dependency": True,
        "temporal_encoder": False,
        "uncertainty_calibration": True,
    },
    "no_calibration": {
        "tensor_factorization": True,
        "spatial_regularization": True,
        "adaptive_dependency": True,
        "temporal_encoder": True,
        "uncertainty_calibration": False,
    },
}
