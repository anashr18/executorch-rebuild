package com.example.xnndemo

import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private fun copyAssetToFiles(assetName: String): String {
        val outFile = File(filesDir, assetName)
        assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        return outFile.absolutePath
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val tv: TextView = findViewById(R.id.tv)

        try {
            // 1) Copy model to a real path
            val ptePath: String = copyAssetToFiles("model.pte")

            // 2) Load module
            val t0: Long = SystemClock.elapsedRealtimeNanos()
            val module: Module = Module.load(ptePath)
            val tLoad: Long = SystemClock.elapsedRealtimeNanos()

            // 3) Create input (match your export: 1x64x1024 FP32)
            val shape: LongArray = longArrayOf(1, 64, 1024)
            val data: FloatArray = FloatArray(1 * 64 * 1024) { 0.1f }
            val input: Tensor = Tensor.fromBlob(data, shape)

            // 4) Forward
            val tRun0: Long = SystemClock.elapsedRealtimeNanos()
            val out: Array<EValue> = module.forward(EValue.from(input))  // or module.forward(*arrayOf(EValue.from(input)))
            val tRun1: Long = SystemClock.elapsedRealtimeNanos()

// 5) Read output (avoid nullable Float?)
            val outT: Tensor = out[0].toTensor()
            val arr: FloatArray = outT.dataAsFloatArray
                ?: error("Expected float32 output; got different dtype or empty tensor")
            val outFirst: Float = arr[0]
            val outShape: String = outT.shape().joinToString("x")


            val loadMs = (tLoad - t0) / 1_000_000.0
            val runMs  = (tRun1 - tRun0) / 1_000_000.0

            tv.text = "✅ Loaded OK\n" +
                    "Load: %.3f ms\n".format(loadMs) +
                    "Forward: %.3f ms\n".format(runMs) +
                    "out[0]=%.4f\n".format(outFirst) +
                    "shape=$outShape\n" +
                    "path=$ptePath"
        } catch (e: Throwable) {
            tv.text = "❌ Error: ${e.message}\n${Log.getStackTraceString(e)}"
        }
    }
}
