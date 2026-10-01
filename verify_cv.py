import cv2
import numpy as np

print("=== 1. VERIFY OPENCV / MAGSAC CLAIM ===")
print("cv2.__version__:", cv2.__version__)
print("hasattr(cv2, 'USAC_MAGSAC'):", hasattr(cv2, 'USAC_MAGSAC'))
print("hasattr(cv2, 'USAC_DEFAULT'):", hasattr(cv2, 'USAC_DEFAULT'))

src = np.array([[0,0], [1,0], [0,1], [1,1]], dtype=np.float32)
dst = np.array([[1,1], [2,1], [1,2], [2,2]], dtype=np.float32)

try:
    if hasattr(cv2, 'USAC_MAGSAC'):
        M, mask = cv2.estimateAffine2D(src, dst, method=cv2.USAC_MAGSAC)
        print("cv2.estimateAffine2D with USAC_MAGSAC: SUCCESS")
    else:
        print("cv2.estimateAffine2D with USAC_MAGSAC: UNAVAILABLE_ENVIRONMENT")
except Exception as e:
    print("cv2.estimateAffine2D with USAC_MAGSAC: UNSUPPORTED_BY_API", e)

try:
    if hasattr(cv2, 'USAC_MAGSAC'):
        H, mask = cv2.findHomography(src, dst, method=cv2.USAC_MAGSAC)
        print("cv2.findHomography with USAC_MAGSAC: SUCCESS")
    else:
        print("cv2.findHomography with USAC_MAGSAC: UNAVAILABLE_ENVIRONMENT")
except Exception as e:
    print("cv2.findHomography with USAC_MAGSAC: UNSUPPORTED_BY_API", e)
    
try:
    if hasattr(cv2, 'USAC_MAGSAC'):
        M, mask = cv2.estimateAffinePartial2D(src[:3], dst[:3], method=cv2.USAC_MAGSAC)
        print("cv2.estimateAffinePartial2D with USAC_MAGSAC: SUCCESS")
    else:
        print("cv2.estimateAffinePartial2D with USAC_MAGSAC: UNAVAILABLE_ENVIRONMENT")
except Exception as e:
    print("cv2.estimateAffinePartial2D with USAC_MAGSAC: UNSUPPORTED_BY_API", e)
