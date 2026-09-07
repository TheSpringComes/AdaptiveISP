"""Samsung Modular Neural ISP operator wrappers.

Import-time side effects register every operator into `isp.registry`.
"""
from isp.learned.samsung_modular import denoise    # noqa: F401
from isp.learned.samsung_modular import awb        # noqa: F401
from isp.learned.samsung_modular import gain       # noqa: F401
from isp.learned.samsung_modular import gtm        # noqa: F401
from isp.learned.samsung_modular import chroma     # noqa: F401
from isp.learned.samsung_modular import gamma      # noqa: F401
from isp.learned.samsung_modular import detail     # noqa: F401

__all__: list[str] = []
