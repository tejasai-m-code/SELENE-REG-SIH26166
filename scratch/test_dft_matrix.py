import numpy as np

def upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    im2pi = 1j * 2.0 * np.pi
    if axis_offsets is None:
        axis_offsets = [0, 0]
    # For inverse DFT matrix multiplication (cross-correlation peak):
    # Notice the +im2pi for IFFT
    t_row = (np.arange(upsampled_region_size[0]) - axis_offsets[0])[:, None] * np.fft.fftfreq(data.shape[0])[None, :]
    kern_row = np.exp(im2pi * t_row / upsample_factor)
    
    t_col = (np.arange(upsampled_region_size[1]) - axis_offsets[1])[:, None] * np.fft.fftfreq(data.shape[1])[None, :]
    kern_col = np.exp(im2pi * t_col / upsample_factor)
    
    return kern_row @ data @ kern_col.T

h, w = 128, 128
rng = np.random.default_rng(42)
src = rng.normal(0, 1, (h, w))

shift_y, shift_x = 0.35, -0.42
fy = np.fft.fftfreq(h)[:, None]
fx = np.fft.fftfreq(w)[None, :]
phase = np.exp(-2j * np.pi * (fy * shift_y + fx * shift_x))
ref = np.fft.ifft2(np.fft.fft2(src) * phase).real

# If ref(y, x) = src(y - shift_y, x - shift_x)
# F_ref * conj(F_src) has peak at (shift_y, shift_x)
F_ref = np.fft.fft2(ref)
F_src = np.fft.fft2(src)
prod = F_ref * np.conj(F_src)
prod /= np.abs(prod)

upsample_factor = 50
ups_size = int(np.ceil(upsample_factor * 1.5))
dftshift = int(np.trunc(ups_size / 2.0))

cc = np.fft.ifft2(prod)
maxima = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
shifts = np.array(maxima, dtype=float)
midpoints = np.array([h / 2.0, w / 2.0])
for k in range(2):
    if shifts[k] > midpoints[k]:
        shifts[k] -= [h, w][k]

row_offset = dftshift - shifts[0] * upsample_factor
col_offset = dftshift - shifts[1] * upsample_factor

CC = upsampled_dft(prod, (ups_size, ups_size), upsample_factor, (row_offset, col_offset))
up_max = np.unravel_index(np.argmax(np.abs(CC)), CC.shape)
up_shifts = (np.array(up_max, dtype=float) - dftshift) / upsample_factor
total_shift = shifts + up_shifts
print(f"True shift (y, x): {shift_y:.4f}, {shift_x:.4f}")
print(f"Recovered shift (y, x): {total_shift[0]:.4f}, {total_shift[1]:.4f}")
print(f"Error (y, x): {abs(total_shift[0] - shift_y):.6f}, {abs(total_shift[1] - shift_x):.6f}")
