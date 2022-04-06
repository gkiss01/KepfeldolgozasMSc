import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.highgui.HighGui
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import kotlin.system.exitProcess

const val IMAGE_PROCESSING_NUMBER_OF_PARTS = 8

fun main() {
    OpenCV.loadLocally()

    initApplicationWindows()

    val cap = VideoCapture(0, Videoio.CAP_DSHOW)
    if (cap.isOpened) {
        val img = Mat()
        while (true) {
            if (!cap.read(img)) continue
            processImage(img)

            val keyPressed = HighGui.waitKey(16)
            if (keyPressed == 27) break
        }
    }

    cap.release()

    HighGui.waitKey()
    HighGui.destroyAllWindows()

    exitProcess(0)
}

fun processImage(
    src: Mat,
): Double {
    src.resizeIfNeeded(sizeOfImage.width.toInt(), sizeOfImage.height.toInt())
    Core.flip(src, src, 1)
    HighGui.imshow(WINDOW_NAME_ORIGINAL, src)

    val handImg = keepHand(src)
    HighGui.imshow(WINDOW_NAME_PROCESSED, handImg)

    val splitImgData = splitImage(handImg, IMAGE_PROCESSING_NUMBER_OF_PARTS)
    highlightImagePart(splitImgData.splitImage, splitImgData.largestPartIndex, splitImgData.imagePartsData.size)
    HighGui.imshow(WINDOW_NAME_SPLIT, splitImgData.splitImage)

    val arrowImg = createArrowImage(splitImgData.asAngle)
    arrowImg.resizeIfNeeded(sizeOfImage.width.toInt(), sizeOfImage.height.toInt())
    HighGui.imshow(WINDOW_NAME_ARROW, arrowImg)

    return splitImgData.asAngle
}

fun keepHand(
    src: Mat,
    lowerColor: DoubleArray = doubleArrayOf(90.0, 105.0, 0.0),
    upperColor: DoubleArray = doubleArrayOf(110.0, 230.0, 255.0)
): Mat {
    val dst = Mat()

    Imgproc.GaussianBlur(src, dst, Size(3.0, 3.0), 0.0)
    Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2HSV)
    Core.inRange(dst, Scalar(lowerColor), Scalar(upperColor), dst)

    Imgproc.medianBlur(dst, dst, 5)
    Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(8.0, 8.0)).apply {
        Imgproc.dilate(dst, dst, this)
    }

    return dst
}

fun splitImage(
    src: Mat,
    parts: Int = 3,
    partColors: List<Scalar> = generateColorsInHsv(parts)
): SplitImageData {

    if (partColors.size < parts) return SplitImageData(Mat(), emptyList())

    val dst = Mat()
    val masks = generateImagePartMasks(src, parts)

    val imagePartsData = mutableListOf<ImagePartData>()
    val totalPixels = Core.countNonZero(src)

    Imgproc.cvtColor(src, dst, Imgproc.COLOR_GRAY2BGR)
    Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2HSV)

    for (j in 0 until parts) {
        val partImg = Mat().apply { src.copyTo(this, masks[j]) }
        dst.setTo(partColors[j], partImg)

        val partPixels = Core.countNonZero(partImg)
        imagePartsData.add(ImagePartData(j, partPixels / totalPixels.toDouble()))
    }

    Imgproc.cvtColor(dst, dst, Imgproc.COLOR_HSV2BGR)

    return SplitImageData(dst, imagePartsData)
}

fun highlightImagePart(
    src: Mat,
    part: Int,
    totalParts: Int,
    color: Scalar = Scalar(0.0, 69.0, 255.0)
) {
    if (part < 0 || part > totalParts) return

    val rects = generateImagePartRects(src, totalParts)
    Imgproc.rectangle(src, rects[part], color, 2)
}

fun generateImagePartMasks(
    src: Mat,
    parts: Int
): List<Mat> {
    val masks = mutableListOf<Mat>()
    val rects = generateImagePartRects(src, parts)

    for (i in 0 until parts) {
        val mask = Mat.zeros(src.size(), CvType.CV_8UC1)
        Imgproc.rectangle(mask, rects[i], Scalar(255.0, 255.0, 255.0), -1)
        masks.add(mask)
    }

    return masks
}

fun generateImagePartRects(
    src: Mat,
    parts: Int
): List<Rect> {
    val rects = mutableListOf<Rect>()
    val intervals = splitDiscreteInterval(0, src.cols() - 1, parts)

    for (i in 0 until parts) {
        val point1 = Point(intervals[i].first.toDouble(), 0.0)
        val point2 =
            Point(intervals[i].second.toDouble() + 1, src.height().toDouble()) // délkelet irányába 1 pixelnyi eltolás
        rects.add(Rect(point1, point2))
    }

    return rects
}

fun createArrowImage(
    angle: Double,
    paintColor: Scalar = Scalar(64.0, 64.0, 64.0),
    backgroundColor: Scalar = Scalar(255.0, 255.0, 255.0)
): Mat {
    val img = Mat(Size(600.0, 600.0), CvType.CV_8UC3, backgroundColor)
    val start = Point(300.0, 300.0)
    val end = Point(500.0, 300.0)

    if (angle.isNaN()) Imgproc.circle(img, start, 200, paintColor, 10)
    else Imgproc.arrowedLine(img, start, end.rotate(start, Math.toRadians(-angle)), paintColor, 10)

    return img
}