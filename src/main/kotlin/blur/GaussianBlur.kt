package blur

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

const val IMAGE_BLUR_BORDER_PADDING = 25

fun gaussianBlurColored(
    src: Mat,
    ksize: Int = 5
): Mat {
    // a kép csatornákra bontása => szürkeslálás képek halmaza
    val channels: MutableList<Mat> = ArrayList()
    Core.split(src, channels)

    // elmosás minden egyes csatornán
    for (i in 0 until channels.size)
        channels[i] = gaussianBlur(channels[i], ksize)

    // színes kép visszanyerése a csatornák összefűzésével
    val result = Mat()
    Core.merge(channels, result)

    return result
}

fun gaussianBlur(
    src: Mat,
    ksize: Int = 5
): Mat {
    // az előkészítés változtatja a kép méretét, ezért azt elmentjük
    val originalSize = src.size()

    // kép előkészítése a Fourier transzformációhoz (két szintre vágás, méretezés)
    val preparedSrc = prepareFourierTransformation(src)

    // Fourier transzformáció
    val complexSrc = fourierTransformation(preparedSrc)

    // szűrő kép (kernel) létrehozása (ugyanolyan méretű kell legyen, mint a preparedSrc)
    val kernel = gaussianKernelEx(ksize, preparedSrc.size())

    // Fourier transzformáció a szűrő képen
    val complexKernel = fourierTransformation(kernel)

    // SZŰRÉS => a transzformált kép és a szűrő kép elemenként történő összeszorzása
    Core.mulSpectrums(complexSrc, complexKernel, complexSrc, 0)

    // inverz Fourier transzformáció
    return inverseFourierTransformation(complexSrc, originalSize)
}

fun prepareFourierTransformation(
    src: Mat
): Mat {
    // az eredeti kép szürkeskálássá alakítása
    var graySrc = Mat()
    if (src.type() == CvType.CV_8U) graySrc = src.clone()
    else Imgproc.cvtColor(src, graySrc, Imgproc.COLOR_BGR2GRAY)

    // a transzformációhoz optimális képméret meghatározása (a gyorsaság miatt szükséges)
    val m = Core.getOptimalDFTSize(graySrc.rows())
    val n = Core.getOptimalDFTSize(graySrc.cols())

    // a szegélyen lévő új elemek kitöltése 0 értékekkel
    val paddedSrc = Mat()
    Core.copyMakeBorder(
        graySrc,
        paddedSrc,
        IMAGE_BLUR_BORDER_PADDING,
        m - graySrc.rows() + IMAGE_BLUR_BORDER_PADDING,
        IMAGE_BLUR_BORDER_PADDING,
        n - graySrc.cols() + IMAGE_BLUR_BORDER_PADDING,
        Core.BORDER_REPLICATE
    )

    return paddedSrc
}

fun fourierTransformation(
    preparedSrc: Mat
): Mat {
    // a Fourier transzformáció eredménye komplex kép => komplex komponensek tárolása szükséges
    val planes: MutableList<Mat> = ArrayList()
    preparedSrc.convertTo(preparedSrc, CvType.CV_32F)
    planes.add(preparedSrc)
    planes.add(Mat.zeros(preparedSrc.size(), CvType.CV_32F))

    // komplex kép összeállítása
    val complexSrc = Mat()
    Core.merge(planes, complexSrc)

    // Fourier transzformáció
    Core.dft(complexSrc, complexSrc)
    return complexSrc
}

fun inverseFourierTransformation(
    complexSrc: Mat,
    originalSize: Size
): Mat {
    // inverz Fourier transzformáció
    Core.idft(complexSrc, complexSrc)

    // komplex komponensek kiemelése
    val planes: MutableList<Mat> = ArrayList()
    Core.split(complexSrc, planes)

    // valós értékek (tagok) normalizálása
    var restoredSrc = Mat()
    Core.normalize(planes[0], restoredSrc, 0.0, 255.0, Core.NORM_MINMAX)

    // az előkészítés során hozzáadott szegélyek eldobása
    restoredSrc = restoredSrc.submat(
        Rect(
            IMAGE_BLUR_BORDER_PADDING,
            IMAGE_BLUR_BORDER_PADDING,
            originalSize.width.toInt(),
            originalSize.height.toInt()
        )
    )

    // a normalizált kép visszalakaítása szürkeskálássá
    restoredSrc.convertTo(restoredSrc, CvType.CV_8U)

    return restoredSrc
}

fun gaussianKernel(
    ksize: Int = 3
): Mat {
    // kérünk egy megfelelő méretű 1D-s kernelt
    val kernel = Imgproc.getGaussianKernel(ksize, 0.0) // a szigma a 'ksize' alapján kerül meghatározásra

    // kernel transzponálása
    val kernelT = Mat()
    Core.transpose(kernel, kernelT)

    // a két mátrix összeszorzása => 2D-s kernel meghatározása
    val product = Mat()
    Core.gemm(kernel, kernelT, 1.0, Mat(), 0.0, product, 0)
    return product
}

fun gaussianKernelEx(
    ksize: Int,
    size: Size
): Mat {
    // kérünk egy megfelelő méretű 2D-s kernelt
    val kernel = gaussianKernel(ksize)

    // kernel beágyazása egy megfelelő méretű nagyobb kép bal felső sarkába
    val kernelEx = Mat()
    Core.copyMakeBorder(
        kernel,
        kernelEx,
        0,
        size.height.toInt() - kernel.rows(),
        0,
        size.width.toInt() - kernel.cols(),
        Core.BORDER_CONSTANT,
        Scalar.all(0.0)
    )

    return kernelEx
}