import nu.pattern.OpenCV
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

fun main(args: Array<String>) {
    OpenCV.loadLocally()

    HighGui.namedWindow("Hand")
    HighGui.resizeWindow("Hand", 200, 200)
    HighGui.moveWindow("Hand", 100, 100)

    var img: Mat
    for (i in 0..2)
        for (j in 0..49) {
            img = Imgcodecs.imread("./TestPictures/Dataset2/train/$i/$j.png", Imgcodecs.IMREAD_COLOR)
            val handImg = keepHand(img)

            HighGui.imshow("Hand", handImg)
            HighGui.waitKey()
        }

    HighGui.waitKey()
    HighGui.destroyAllWindows()
}

fun keepHand(src: Mat): Mat {
    val lowerColor = doubleArrayOf(0.0, 58.0, 50.0)
    val upperColor = doubleArrayOf(30.0, 255.0, 255.0)

    val blur = Mat()
    val hsv = Mat()
    val mask = Mat()

    Imgproc.GaussianBlur(src, blur, Size(3.0, 3.0), 0.0)
    Imgproc.cvtColor(blur, hsv, Imgproc.COLOR_BGR2HSV)
    Core.inRange(hsv, Scalar(lowerColor), Scalar(upperColor), mask)

    Imgproc.medianBlur(mask, blur, 5)
    val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(8.0, 8.0))
    Imgproc.dilate(blur, blur, kernel)

    return blur
}