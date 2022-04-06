import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.highgui.HighGui
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import kotlin.system.exitProcess

// hány partícióra bontsuk szét a képet a statisztika elkészítésekor
// a magasabb érték pontosabb szögszámítást eredményez, de erőforrásigényesebb
const val IMAGE_PROCESSING_NUMBER_OF_PARTS = 8

fun main() {
    // OpenCV betöltése
    OpenCV.loadLocally()

    // alkalmazásablakokok létrehozása
    initApplicationWindows()

    // videó bemenet létrehozása
    val cap = VideoCapture(0, Videoio.CAP_DSHOW)
    if (cap.isOpened) { // ha sikerült kapcsolódni a kamerához, kezdjük meg a feldolgozást
        val img = Mat()
        while (true) {
            if (!cap.read(img)) continue // a sikertelen képkockákat eldobjuk
            processImage(img) // kép feldolgozása

            val keyPressed = HighGui.waitKey(16) // 16 ms késeltetés (60 fps)
            if (keyPressed == 27) break // ESC billentyű lenyomásakor a feldolgozás leállítása
        }
    }

    // kamera erőforrás elengedése
    cap.release()

    // összes ablak törlése
    HighGui.waitKey()
    HighGui.destroyAllWindows()

    // folyamat megszüntetése
    exitProcess(0)
}

fun processImage( // a teljes feldolgozást és megjelenítést összefogó függvény
    src: Mat,
): Double {
    // kamerától érkező kép méretezése (az arányokat megtartva!), amennyiben szükséges
    src.resizeIfNeeded(sizeOfImage.width.toInt(), sizeOfImage.height.toInt())
    // a kamerától érkező kép alapból tükrözve érkezik, így manuálisan visszatükrözzük
    Core.flip(src, src, 1)
    HighGui.imshow(WINDOW_NAME_ORIGINAL, src) // megjelenítés a megfelelő ablakban

    // a kezet tartalmazó maszk létrehozása
    val handImg = keepHand(src)
    HighGui.imshow(WINDOW_NAME_PROCESSED, handImg) // megjelenítés

    // feldarabolt kép és statisztika kinyerése a maszkból
    val splitImgData = splitImage(handImg, IMAGE_PROCESSING_NUMBER_OF_PARTS)
    // legnagyobb intenzitású partíció kiemelése a feldarabolt képen
    highlightImagePart(splitImgData.splitImage, splitImgData.largestPartIndex, splitImgData.imagePartsData.size)
    HighGui.imshow(WINDOW_NAME_SPLIT, splitImgData.splitImage) // megjelenítés

    // eredmény kép létrehozása a statisztikából számolt szög alapján
    val arrowImg = createArrowImage(splitImgData.asAngle)
    // eredmény kép újraméretezése, hogy egyezzen a feldolgozott kép méretével (esztétikai okok miatt)
    arrowImg.resizeIfNeeded(sizeOfImage.width.toInt(), sizeOfImage.height.toInt())
    HighGui.imshow(WINDOW_NAME_ARROW, arrowImg) // megjelenítés

    return splitImgData.asAngle
}

fun keepHand( // a kezet tartalmazó maszk létrehozása
    src: Mat,
    lowerColor: DoubleArray = doubleArrayOf(90.0, 105.0, 0.0),
    upperColor: DoubleArray = doubleArrayOf(110.0, 230.0, 255.0)
): Mat {
    val dst = Mat()

    // bemeneti kép enyhe elmosása
    Imgproc.GaussianBlur(src, dst, Size(3.0, 3.0), 0.0)

    // átváltunk HSV színtérre, majd egy maszkot készítünk a tartományba eső pontokról
    Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2HSV)
    Core.inRange(dst, Scalar(lowerColor), Scalar(upperColor), dst)

    // maszkon belüli apró lyukak és körvonal menti kis mértékű egyenetlenségek kiküszöbölése
    Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(8.0, 8.0)).apply {
        Imgproc.dilate(dst, dst, this)
    }

    // a belül esetlegesen megmaradt nagy lyukak érdességének csökkentése
    Imgproc.medianBlur(dst, dst, 11)

    return dst
}

fun splitImage( // a kezet tartalmazó kép darabolása, a feldolgozott kép előállítása
    src: Mat,
    parts: Int = 3,
    partColors: List<Scalar> = generateColorsInHsv(parts)
): SplitImageData {

    if (partColors.size < parts) throw Exception("Size of partColors is too small!")

    val dst = Mat()

    // létrehozzuk a darabolásához használt maszkokat
    val masks = generateImagePartMasks(src, parts)

    val imagePartsData = mutableListOf<ImagePartData>()
    val totalPixels = Core.countNonZero(src) // összesen hány nem 0 pixel van a képen

    // mivel a kezet tartalmazó bemeneti kép szürkeskálás, átváltjuk színesre (előbb BGR, majd HSV színtér)
    // ez azért szükséges, hogy ki különböző színnel tudjunk rá rajzolni
    Imgproc.cvtColor(src, dst, Imgproc.COLOR_GRAY2BGR)
    Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2HSV)

    for (j in 0 until parts) {
        // a kép maszkolása az egyes partíciók maszkjaival
        val partImg = Mat().apply { src.copyTo(this, masks[j]) }

        // maszkolt kép (partíció) nem 0 pixeleinek megszámolása, majd arány kiszámítása
        val partPixels = Core.countNonZero(partImg)
        imagePartsData.add(ImagePartData(j, partPixels / totalPixels.toDouble()))

        // az eredmény kép aktuálisan feldolgozandó partícióját átszínezzük
        dst.setTo(partColors[j], partImg)
    }

    // visszatérés BGR színtérbe a megjelenítés miatt
    Imgproc.cvtColor(dst, dst, Imgproc.COLOR_HSV2BGR)

    // számolt statisztika visszaadása
    return SplitImageData(dst, imagePartsData)
}

fun highlightImagePart( // adott indexű partíció kiemelése a képen
    src: Mat,
    part: Int,
    totalParts: Int,
    color: Scalar = Scalar(0.0, 69.0, 255.0)
) {
    if (part < 0 || part > totalParts) return // throw Exception("The part number cannot be greater than totalParts.")

    // létrehozzuk a darabolásához használt téglalapokat a megfelelő számban
    val rects = generateImagePartRects(src, totalParts)

    // kiemeljük a kívánt téglalapot a képen
    Imgproc.rectangle(src, rects[part], color, 2)
}

fun generateImagePartMasks( // a kép darabolásához használt maszkok létrehozása
    src: Mat,
    parts: Int
): List<Mat> {
    val masks = mutableListOf<Mat>()

    // létrehozzuk a darabolásához használt téglalapokat
    val rects = generateImagePartRects(src, parts)

    for (i in 0 until parts) {
        // fekete kép (maszk) létrehozása az eredeti méretében
        val mask = Mat.zeros(src.size(), CvType.CV_8UC1)

        // fehér színnel kitöltött téglalap rárajzolása a maszkra
        Imgproc.rectangle(mask, rects[i], Scalar(255.0, 255.0, 255.0), -1)
        masks.add(mask)
    }

    return masks
}

fun generateImagePartRects( // a kép darabolásához használt téglalapok létrehozása
    src: Mat,
    parts: Int
): List<Rect> {
    val rects = mutableListOf<Rect>()

    // diszkrét intervallumokon dolgozunk (pixelszám), ezért a saját függvényünkkel határozzuk meg az intervallumokat
    val intervals = splitDiscreteInterval(0, src.cols() - 1, parts)

    for (i in 0 until parts) {
        // téglalap bal felső koordinátája y, x sorrendben
        val point1 = Point(intervals[i].first.toDouble(), 0.0)

        // téglalap jobb alsó koordinátája y, x sorrendben
        val point2 = Point(
            intervals[i].second.toDouble() + 1, // délkelet irányában 1 pixelnyi eltolás, hogy ne maradjon ki pixel
            src.height().toDouble()
        )
        rects.add(Rect(point1, point2))
    }

    return rects
}

fun createArrowImage( // kimenetként előállított kép létrehozása
    angDeg: Double,
    paintColor: Scalar = Scalar(64.0, 64.0, 64.0),
    backgroundColor: Scalar = Scalar(255.0, 255.0, 255.0)
): Mat {
    // 600x600 méretű "vászon" kép létrehozása
    val img = Mat(Size(600.0, 600.0), CvType.CV_8UC3, backgroundColor)

    val start = Point(300.0, 300.0)
    val end = Point(500.0, 300.0)

    // 0 fok esetén egy kört rajzolunk a vászonra
    if (angDeg.isNaN()) Imgproc.circle(img, start, 200, paintColor, 10)
    // egyébként pedig egy nyilat
    else Imgproc.arrowedLine(img, start, end.rotate(start, Math.toRadians(-angDeg)), paintColor, 10)

    return img
}