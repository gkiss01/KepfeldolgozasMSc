import org.opencv.core.Mat
import kotlin.math.abs

data class ImagePartData(
    val partNumber: Int,
    val partRatio: Double
)

data class SplitImageData(
    val splitImage: Mat,
    val imagePartsData: List<ImagePartData>
) {
    val largestPartIndex: Int
        get() {
            val maxPart = imagePartsData.maxByOrNull { it.partRatio } ?: return -1
            val equalParts = imagePartsData.filter { it.partRatio == maxPart.partRatio }.size
            return if (equalParts != 1) -1
            else maxPart.partNumber
        }

    val asAngle: Double
        get() {
            val largestPartNumber = imagePartsData.size - 1
            val partAngle = 180.0 / largestPartNumber
            var sumAngle = 0.0

            for (imagePartData in imagePartsData) {
                sumAngle += abs(imagePartData.partNumber - largestPartNumber) * imagePartData.partRatio * partAngle
            }
            return sumAngle
        }
}
