package com.example.detect

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private lateinit var cameraView: JavaCameraView
    private lateinit var resultTextView: TextView
    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        //UI 요소 연결
        cameraView = findViewById(R.id.camera_view)
        resultTextView = findViewById(R.id.result_text_view)

        //권한 확인 및 요청
        if (!checkPermission()) {
            requestPermission()
        }

        //OpenCV 초기화
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV Loaded Successfully")
        } else {
            Log.d(TAG, "Failed to load OpenCV")
        }

        //JavaCameraView 설정
        cameraView.visibility = SurfaceView.VISIBLE
        cameraView.setCvCameraViewListener(this)
        cameraView.enableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        Log.d(TAG, "Camera View Started")
    }

    override fun onCameraViewStopped() {
        Log.d(TAG, "Camera View Stopped")
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        val frame = inputFrame.rgba()

        //신호등 인식 및 결과 표시
        val signalStatus = detectTrafficSignal(frame)
        runOnUiThread { resultTextView.text = signalStatus }

        return frame
    }

    private fun detectTrafficSignal(mat: Mat): String {
        //1. 블러 처리로 노이즈 제거
        Imgproc.GaussianBlur(mat, mat, Size(9.0, 9.0), 2.0)

        //2. 그레이스케일 변환
        val grayMat = Mat()
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)

        //3. 원 검출(HoughCircle Transform)
        val circles = Mat()
        Imgproc.HoughCircles(
            grayMat,
            circles,
            Imgproc.HOUGH_GRADIENT,
            1.0,
            grayMat.rows() / 8.0,
            100.0,
            30.0,
            20, // 최소 반지름
            100 // 최대 반지름
        )

        if (circles.cols() > 0) {
            val detectedSignals = mutableListOf<Pair<Point, String>>()

            for (x in 0 until circles.cols()) {
                val circleData = circles.get(0, x) ?: continue
                val center = Point(circleData[0], circleData[1])
                val radius = circleData[2].toInt()

                //크기 및 위치 제한
                if (radius < 20 || radius > 100 || center.y > mat.rows() / 3) {
                    continue
                }

                //ROI(관심영역) 설정
                val roi = Rect(
                    (center.x - radius).toInt().coerceAtLeast(0),
                    (center.y - radius).toInt().coerceAtLeast(0),
                    (2 * radius).toInt().coerceAtMost(mat.cols()),
                    (2 * radius).toInt().coerceAtMost(mat.rows())
                )
                val roiMat = Mat(mat, roi)

                //HSV 변환
                val hsvMat = Mat()
                Imgproc.cvtColor(roiMat, hsvMat, Imgproc.COLOR_BGR2HSV)

                //색상 마스크 생성
                val redMask = Mat()
                val greenMask = Mat()
                val yellowMask = Mat()

                Core.inRange(hsvMat, Scalar(0.0, 100.0, 100.0), Scalar(10.0, 255.0, 255.0), redMask)
                Core.inRange(hsvMat, Scalar(40.0, 100.0, 100.0), Scalar(80.0, 255.0, 255.0), greenMask)
                Core.inRange(hsvMat, Scalar(15.0, 150.0, 150.0), Scalar(35.0, 255.0, 255.0), yellowMask)

                //픽셀 비율 계산
                val totalPixels = Core.countNonZero(Mat.ones(roi.size(), CvType.CV_8U))
                val redRatio = Core.countNonZero(redMask).toDouble() / totalPixels
                val greenRatio = Core.countNonZero(greenMask).toDouble() / totalPixels
                val yellowRatio = Core.countNonZero(yellowMask).toDouble() / totalPixels

                //상태 판단
                val signals = mutableListOf<String>()
                if (redRatio > 0.4) signals.add("STOP")
                if (greenRatio > 0.4) signals.add("GO")
                if (yellowRatio > 0.4) signals.add("WAIT")

                //다중 상태 처리
                return if (signals.size > 1) {
                    signals.joinToString(" & ") // 예: "STOP & WAIT"
                } else if (signals.size == 1) {
                    signals.first()
                } else {
                    "UNKNOWN"
                }
            }
        }

        return "NO SIGNAL"
    }

    private fun checkPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermission() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraView.disableView()
    }
}



