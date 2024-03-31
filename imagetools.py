from collections.abc import Iterable, Sequence
from copy import deepcopy
from operator import index
from os.path import dirname, normpath
from pathlib import Path
from typing import Any, SupportsFloat, SupportsIndex

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image

# open RGBA images up to 1 GiB (16384 x 16384 pixels)
Image.MAX_IMAGE_PIXELS = (1024**3)//4

# 8-bit integer sRGB(A)-convertible
iIMGCompatible = (npt.NDArray[np.integer] |
                  Iterable[Iterable[Iterable[SupportsIndex]]] |
                  Iterable[Iterable[SupportsIndex]] |
                  Iterable[SupportsIndex])

# 64-bit float ([0, 1]) sRGB(A)-convertible
fIMGCompatible = (npt.NDArray[np.floating] |
                  Iterable[Iterable[Iterable[SupportsFloat]]] |
                  Iterable[Iterable[SupportsFloat]] |
                  Iterable[SupportsFloat])

# generic sRGB(A)-convertible
IMGCompatible = iIMGCompatible | fIMGCompatible


def is_iIMGCompatible(OBJ: Any
                      ) -> bool:
    """Perform a rudimentary check of whether an object
    is consistent with the type alias iIMGCompatible.

    Keyword arguments:\n
    OBJ -- An object to check for consistency with iIMGCompatible.
    """
    obj: Any = deepcopy(OBJ)
    if isinstance(obj, np.ndarray):
        if issubclass(obj.dtype.type, np.integer):
            return (True)
        return (False)
    elif isinstance(obj, Iterable):
        for subobj in obj:
            if isinstance(subobj, Iterable):
                for subsubobj in subobj:
                    if isinstance(subsubobj, Iterable):
                        for subsubsubobj in subobj:
                            if isinstance(subsubsubobj, SupportsIndex):
                                return (True)
                            break
                    elif isinstance(subsubobj, SupportsIndex):
                        return (True)
                    break
            elif isinstance(subobj, SupportsIndex):
                return (True)
            break
    return (False)


def is_fIMGCompatible(OBJ: Any
                      ) -> bool:
    """Perform a rudimentary check of whether an object
    is consistent with the type alias fIMGCompatible.

    Keyword arguments:\n
    OBJ -- An object to check for consistency with fIMGCompatible.
    """
    obj: Any = deepcopy(OBJ)
    if isinstance(obj, np.ndarray):
        if issubclass(obj.dtype.type, np.floating):
            return (True)
        return (False)
    elif isinstance(obj, Iterable):
        for subobj in obj:
            if isinstance(subobj, Iterable):
                for subsubobj in subobj:
                    if isinstance(subsubobj, Iterable):
                        for subsubsubobj in subobj:
                            if isinstance(subsubsubobj, SupportsFloat):
                                return (True)
                            break
                    elif isinstance(subsubobj, SupportsFloat):
                        return (True)
                    break
            elif isinstance(subobj, SupportsFloat):
                return (True)
            break
    return (False)


def _iimgify(img: iIMGCompatible
             ) -> npt.NDArray[np.uint8]:
    """Return an 8-bit [0, 255] integer representation of img.

    Keyword arguments:\n
    img -- An integer image-like object to convert to
           8-bit integer [0, 255] (s)RGBA format.
    """
    if is_iIMGCompatible(img):
        out: npt.NDArray[np.uint8] = np.clip(np.array(img),
                                             0, 255).astype(np.uint8)
        if len(out.shape) > 3:
            raise ValueError
        while len(out.shape) < 3:
            out = np.array([out], dtype=np.uint8)
        if out.shape[2] not in [3, 4]:
            raise ValueError
        if out.shape[2] == 3:
            # add trivial alpha-values
            out = np.concatenate((out,
                                  255 * np.ones((out.shape[0],
                                                 out.shape[1],
                                                 1),
                                                dtype=np.uint8)),
                                 axis=2)
        return (np.array(out, dtype=np.uint8))
    else:
        raise TypeError


def _fimgify(img: fIMGCompatible
             ) -> npt.NDArray[np.float64]:
    """Return a 64-bit [0, 1] floating point representation of img.

    Keyword arguments:\n
    img -- A floating-point image-like object to convert to
           64-bit float [0, 1] (s)RGBA format.
    """
    if is_fIMGCompatible(img):
        out: npt.NDArray[np.float64] = np.clip(np.array(img),
                                               0, 1).astype(np.float64)
        if len(out.shape) > 3:
            raise ValueError
        while len(out.shape) < 3:
            out = np.array([out], dtype=np.float64)
        if out.shape[2] not in [3, 4]:
            raise ValueError
        if out.shape[2] == 3:
            # add trivial alpha-values
            out = np.concatenate((out,
                                  np.ones((out.shape[0],
                                           out.shape[1],
                                           1),
                                          dtype=np.float64)),
                                 axis=2)
        return (np.array(out, dtype=np.float64))
    else:
        raise TypeError


def fimg_to_iimg(img: fIMGCompatible
                 ) -> npt.NDArray[np.uint8]:
    """Convert an floating-point image-like object to an 8-bit integer
    [0, 255] representation.

    Keyword arguments:\n
    img -- A floating-point image-like object to convert to
           8-bit integer [0, 255] (s)RGBA format.
    """
    IMG: npt.NDArray[np.float64] = _fimgify(img)
    return (np.array(np.rint(IMG * 255),
                     dtype=np.uint8))


def iimg_to_fimg(img: iIMGCompatible
                 ) -> npt.NDArray[np.float64]:
    """Convert an integer image-like object to a 64-bit floating-point
    [0, 1] representation.

    Keyword arguments:\n
    img -- An integer image-like object to convert to
           64-bit float [0, 1] (s)RGBA format.
    """
    IMG: npt.NDArray[np.uint8] = _iimgify(img)
    return (np.array(IMG.astype(np.float64)/255,
                     dtype=np.float64))


def iimgify(img: IMGCompatible
            ) -> npt.NDArray[np.uint8]:
    """Return a 8-bit [0, 255] integer representation of img.

    Keyword arguments:\n
    img -- An image-like object to convert to 8-bit integer [0, 255]
           (s)RGBA format.
    """
    if is_iIMGCompatible(img):
        IMG: npt.NDArray[np.uint8] = _iimgify(img)  # type: ignore[type-var]
        return (np.array(IMG, dtype=np.uint8))
    elif is_fIMGCompatible(img):
        IMG: npt.NDArray[np.float64] = _fimgify(img)  # type: ignore[type-var]
        return (fimg_to_iimg(IMG))
    else:
        raise TypeError


def fimgify(img: IMGCompatible
            ) -> npt.NDArray[np.float64]:
    """Return a 64-bit [0, 1] floating point representation of img.

    Keyword arguments:\n
    img -- An image-like object to convert to 64-bit float [0, 1]
           (s)RGBA format.
    """
    if is_iIMGCompatible(img):
        IMG: npt.NDArray[np.uint8] = _iimgify(img)  # type: ignore[type-var]
        return (iimg_to_fimg(IMG))
    elif is_fIMGCompatible(img):
        IMG: npt.NDArray[np.float64] = _fimgify(img)  # type: ignore[type-var]
        return (np.array(IMG, dtype=np.float64))
    else:
        raise TypeError


def load(fname: str
         ) -> npt.NDArray[np.uint8]:
    """Return a 64-bit [0, 1] floating point representation of img.

    Keyword arguments:\n
    img -- An image-like object to convert to 64-bit float [0, 1]
           (s)RGBA format.
    """
    return (np.array(Image.open(fname).convert("RGBA"),
                     dtype=np.uint8))


def prepare_save(fname: str
                 ) -> None:
    """Create any necessary parent directories to fname.
    Will not create a file at fname.

    Keyword arguments:\n
    fname -- A string specifying a file to create parent directories of.
    """
    filename: str = normpath(fname)
    Path(dirname(filename)).mkdir(parents=True, exist_ok=True)


def save(img: IMGCompatible,
         fname: str
         ) -> None:
    """Save an image to a file.
    Creates any necessary parent directories in the process.

    Keyword arguments:\n
    img -- An image-like object to save to the file fname.
    fname -- A string specifying the location to save img to.
    """
    prepare_save(fname)
    plt.imsave(fname, iimgify(img))


def save_pdf(img: iIMGCompatible,
             fname: str,
             bg_clr: Sequence[SupportsIndex] = [255, 255, 255]
             ) -> None:
    """Save an image to a file as a PDF.
    Alpha-composites img with background colour bg_clr before saving.
    Creates any necessary parent directories in the process.
    """
    prepare_save(fname)
    IMG: npt.NDArray[np.uint8] = _iimgify(img)
    image = Image.fromarray(IMG, mode="RGBA")
    rgb_img = Image.new("RGB",
                        image.size,
                        (index(bg_clr[0]),
                         index(bg_clr[1]),
                         index(bg_clr[2])))
    rgb_img.paste(image, mask=image.split()[3])
    rgb_img.save(fname, format="PDF")


def save_gif(imgs: Sequence[IMGCompatible],
             fname: str,
             fps: SupportsIndex = 24
             ) -> None:
    """Save a sequence of images to a file as a GIF.
    Creates any necessary parent directories in the process.
    """
    frames: list[npt.NDArray[np.uint8]] = [iimgify(img)
                                           for img in imgs]
    frame0: Image.Image = Image.fromarray(frames[0], mode="RGBA")
    frame0.save(fname,
                format="GIF",
                save_all=True,
                append_images=[Image.fromarray(frame)
                               for frame in frames[1:]],
                duration=1000//index(fps),
                disposal=2,
                loop=0)


def save_apng(imgs: Sequence[IMGCompatible],
              fname: str,
              fps: SupportsIndex = 32
              ) -> None:
    """Save a sequence of images to a file as an APNG.
    Creates any necessary parent directories in the process.
    """
    frames: list[npt.NDArray[np.uint8]] = [iimgify(img)
                                           for img in imgs]
    frame0 = Image.fromarray(frames[0], mode="RGBA")
    frame0.save(fname,
                format="PNG",
                optimize=True,
                save_all=True,
                default_image=True,
                append_images=[Image.fromarray(frame)
                               for frame in frames],
                duration=1000//index(fps),
                disposal=1,
                blend=0,
                loop=0)


def rmclr(img: iIMGCompatible,
          clr: Sequence[SupportsIndex] = [255, 255, 255, 255]
          ) -> npt.NDArray[np.uint8]:
    """Remove all instances of a given colour clr from an image.
    The standard use case is to remove a monochromatic background.
    """
    IMG: npt.NDArray[np.uint8] = _iimgify(img)
    if len(clr) == 3:
        msk = ((IMG[:, :, 0] == index(clr[0])) &
               (IMG[:, :, 1] == index(clr[1])) &
               (IMG[:, :, 2] == index(clr[2])))
    elif len(clr) == 4:
        msk = ((IMG[:, :, 0] == index(clr[0])) &
               (IMG[:, :, 1] == index(clr[1])) &
               (IMG[:, :, 2] == index(clr[2])) &
               (IMG[:, :, 3] == index(clr[3])))
    else:
        raise ValueError
    out: npt.NDArray[np.uint8] = IMG.copy()
    out[msk, :] = 0
    return (np.array(out, dtype=np.uint8))


def rmgr(img: IMGCompatible
         ) -> npt.NDArray[np.uint8]:
    """Remove all shades of grey from img."""
    IMG: npt.NDArray[np.float64] = fimgify(img)
    msk = ((IMG[:, :, 0] == IMG[:, :, 1]) &
           (IMG[:, :, 1] == IMG[:, :, 2]))
    out: npt.NDArray[np.uint8] = fimg_to_iimg(IMG)
    out[msk, :] = 0
    return (np.array(out, dtype=np.uint8))


def invert(img: IMGCompatible
           ) -> npt.NDArray[np.uint8]:
    IMG: npt.NDArray[np.float64] = fimgify(img)
    out: npt.NDArray[np.float64] = IMG.copy()
    out[:, :, :-1] = 1 - IMG[:, :, :-1]
    return (fimg_to_iimg(out))


def downscale(img: iIMGCompatible,
              s: SupportsIndex = 1
              ) -> npt.NDArray[np.uint8]:
    """WARNING: Currently does not support floating-point images."""
    scale: int = index(s)
    IMG: npt.NDArray[np.uint8] = _iimgify(img)
    if (IMG.shape[0] % scale) or (IMG.shape[1] % scale):
        print('Bad scale! defaulting to 1.')
        scale = 1
    out: npt.NDArray[np.uint64] = np.zeros((IMG.shape[0]//scale,
                                            IMG.shape[1]//scale,
                                            IMG.shape[2]),
                                           dtype=np.uint64)
    I, J = np.mgrid[0:scale,
                    0:scale]
    for i in range(IMG.shape[0]//scale):
        for j in range(IMG.shape[1]//scale):
            out[i, j, 0] += np.sum(IMG[i*scale + I, j*scale + J, 0])
            out[i, j, 1] += np.sum(IMG[i*scale + I, j*scale + J, 1])
            out[i, j, 2] += np.sum(IMG[i*scale + I, j*scale + J, 2])
            out[i, j, 3] += np.sum(IMG[i*scale + I, j*scale + J, 3])
    out = np.clip(np.rint(out/(scale**2)), 0, 255)
    return (np.array(out, dtype=np.uint8))


def upscale(img: iIMGCompatible,
            s: SupportsIndex = 1
            ) -> npt.NDArray[np.uint8]:
    """WARNING: Currently does not support floating-point images."""
    scale = index(s)
    IMG = _iimgify(img)
    if (scale < 1) or (scale != index(scale)):
        print('Bad scale! defaulting to 1.')
        scale = 1
    out: npt.NDArray[np.uint8] = np.zeros((IMG.shape[0]*scale,
                                           IMG.shape[1]*scale,
                                           IMG.shape[2]),
                                          dtype=np.uint8)
    I, J = np.mgrid[0:IMG.shape[0]*scale,
                    0:IMG.shape[1]*scale]
    out[I, J, 0] = IMG[I//scale, J//scale, 0]
    out[I, J, 1] = IMG[I//scale, J//scale, 1]
    out[I, J, 2] = IMG[I//scale, J//scale, 2]
    out[I, J, 3] = IMG[I//scale, J//scale, 3]
    return (np.array(out, dtype=np.uint8))


def crop_vertical(img: iIMGCompatible
                  ) -> npt.NDArray[np.uint8]:
    IMG = _iimgify(img)
    S = np.where(np.sum(np.sum(IMG, axis=2,
                               dtype=np.uint16),
                        axis=1,
                 dtype=np.uint64))
    out = IMG[np.min(S):np.max(S)+1, :, :]
    return (np.array(out, dtype=np.uint8))


def crop_horizontal(img: iIMGCompatible
                    ) -> npt.NDArray[np.uint8]:
    IMG = _iimgify(img)
    S = np.where(np.sum(np.sum(IMG, axis=2,
                               dtype=np.uint16),
                        axis=0,
                 dtype=np.uint64))
    out = IMG[:, np.min(S):np.max(S)+1, :]
    return (np.array(out, dtype=np.uint8))


def crop(img: iIMGCompatible
         ) -> npt.NDArray[np.uint8]:
    return (crop_horizontal(crop_vertical(img)))


def glue(image0: iIMGCompatible,
         image1: iIMGCompatible,
         gap: SupportsIndex = 0,
         grow: bool = False
         ) -> npt.NDArray[np.uint8]:
    img0 = _iimgify(image0)
    img1 = _iimgify(image1)
    if (abs(img0.shape[0] - img1.shape[0]) % 2) and grow:
        img0 = upscale(image0, 2)
        img1 = upscale(image1, 2)
    y0, x0, _ = img0.shape
    y1, x1, _ = img1.shape
    space00 = np.zeros((max(0, (y1 - y0)//2), x0, 4),
                       dtype=np.uint8)
    space01 = np.zeros((max(0, y1 - y0) - max(0, (y1 - y0)//2), x0, 4),
                       dtype=np.uint8)
    space10 = np.zeros((max(0, (y0 - y1)//2), x1, 4),
                       dtype=np.uint8)
    space11 = np.zeros((max(y0 - y1, 0) - max(0, (y0 - y1)//2), x1, 4),
                       dtype=np.uint8)
    between = np.zeros((max(y0, y1), gap, 4),
                       dtype=np.uint8)
    out = np.concatenate((np.concatenate((space00, img0, space01),
                                         axis=0),
                          between,
                          np.concatenate((space10, img1, space11),
                                         axis=0)),
                         axis=1)
    if np.array(image1, dtype=np.uint8).shape[-1] == 3:
        if np.array(image1, dtype=np.uint8).shape[-1] == 3:
            out = out[:, :, :3]
    return (np.array(out, dtype=np.uint8))


def sRGB_to_linear(img: IMGCompatible
                   ) -> npt.NDArray[np.float64]:
    """Return a linearized (gamma-expanded) form of an sRGB(A) image.

    Technically this is a linearization function for a pseudo-sRGB
    space with A = 0.0625 (which fits the gamma = 2.2 curve slightly
    better than the standard 0.055) but it doesn't really matter; the
    maximum difference between the standard linearization curve and
    this one is less than 1/256 (i.e. nonexistent for 8-bit colour), the
    linearization/delinearization function for this space is C^1-smooth
    (unlike the IEC 61966-2-1:1999 standard), and the math is nicer.
    """
    IMG: npt.NDArray[np.float64] = fimgify(img)
    out: npt.NDArray[np.float64] = IMG.copy()
    out = np.where(out < 5/112,
                   out * (0x5d2b330cd24d3/(1 << 54)),
                   ((1 + (16 * out))/17)**(12/5))
    out[:, :, -1] = IMG[:, :, -1]
    return (np.clip(out, 0, 1).astype(np.float64))


def linear_to_sRGB(img: fIMGCompatible
                   ) -> npt.NDArray[np.float64]:
    """Return a delinearized (gamma-corrected) form of an sRGB(A) image.

    Technically this is a delinearization function for a pseudo-sRGB
    space with A = 0.0625 (which fits the gamma = 2.2 curve slightly
    better than the standard 0.055) but it doesn't really matter; the
    maximum difference between the standard linearization curve and
    this one is less than 1/256 (i.e. nonexistent for 8-bit colour), the
    linearization/delinearization function for this space is C^1-smooth
    (unlike the IEC 61966-2-1:1999 standard), and the math is nicer.
    """
    IMG: npt.NDArray[np.float64] = _fimgify(img)
    out: npt.NDArray[np.float64] = IMG.copy()
    out = np.where(out < 0x10a3248b6eb25b/(1 << 60),
                   out * 0x57ed2dc5406a7/(1 << 47),
                   ((17 * out**(5/12)) - 1)/16)
    out[:, :, -1] = IMG[:, :, -1]
    return (np.clip(out, 0, 1).astype(np.float64))


def linearize(img: iIMGCompatible
              ) -> npt.NDArray[np.float64]:
    """Alias for sRGB_to_linear(img)"""
    return (sRGB_to_linear(img))


def delinearize(img: fIMGCompatible
                ) -> npt.NDArray[np.uint8]:
    """Alias for fimg_to_iimg(sRGB_to_linear(img))"""
    return (fimg_to_iimg(linear_to_sRGB(img)))


def dull(img: iIMGCompatible
         ) -> npt.NDArray[np.uint8]:
    """Dull an image by pulling all of its sRGB values closer to its
    average grey.

    NOT IN RELEASE VERSION YET, AND SHOULD NOT BE UNTIL IT USES THE
        LINEAR CONVERSION FUNCTIONS"""
    IMG = _iimgify(img).astype(np.uint16)
    grey = _iimgify(np.stack((np.mean(IMG, axis=2),)*4, axis=-1)
                    * np.ones(IMG.shape)).astype(np.uint16)
    out = (grey + IMG)//2
    out[:, :, -1] = IMG[:, :, -1]
    return (np.array(out, dtype=np.uint8))
