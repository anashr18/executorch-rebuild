package com.example.etbackendbench

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.facebook.soloader.SoLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class MainActivity : ComponentActivity() {

    companion object {
        // We’ll always push the active model here (XNN or Vulkan)
        private const val MODEL_PATH = "/data/local/tmp/bench/conv_224_xnn.pte"
        private const val NUM_RUNS = 1000
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Important: initialize SoLoader so native libs can be loaded
        SoLoader.init(this, /* native exopackage = */ false)

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    BenchmarkScreen(
                        onRunBenchmark = { updateText ->
                            // Use lifecycleScope so we don’t block UI
                            lifecycleScope.launch {
                                updateText("Running benchmark…")
                                val (avgMs, totalMs) = runBenchmark()
                                updateText(
                                    "Backend model @ $MODEL_PATH\n" +
                                            "Runs: $NUM_RUNS\n" +
                                            "Total: %.3f ms\n".format(totalMs) +
                                            "Avg: %.6f ms / run".format(avgMs)
                                )
                            }
                        }
                    )
                }
            }
        }
    }

    private suspend fun runBenchmark(): Pair<Double, Double> =
        withContext(Dispatchers.Default) {
            // 1. Load the ExecuTorch module from device path
            val module = Module.load(MODEL_PATH)

            // 2. Prepare a dummy input tensor once: shape [1, 1024]
            val inputData = FloatArray(1024) { 1.0f }
            val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 1024))
//            val inputData = FloatArray(65_536) { 1.0f }
//            val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 65_536))

            val eInput = EValue.from(inputTensor)

            // 3. Warm-up runs
            repeat(5) {
                module.forward(eInput)
            }

            // 4. Timed runs
            val t0 = System.nanoTime()
            repeat(NUM_RUNS) {
                module.forward(eInput)
            }
            val t1 = System.nanoTime()

            val totalMs = (t1 - t0) / 1e6      // ns → ms
            val avgMs = totalMs / NUM_RUNS
            Pair(avgMs, totalMs)
        }
}

@Composable
private fun BenchmarkScreen(
    onRunBenchmark: (updateText: (String) -> Unit) -> Unit
) {
    var resultText by remember { mutableStateOf("Press the button to run benchmark.") }

    Column(modifier = Modifier
        .fillMaxSize()
        .padding(24.dp)
    ) {
        Button(onClick = { onRunBenchmark { newText -> resultText = newText } }) {
            Text("Run benchmark")
        }

        Text(
            text = resultText,
            modifier = Modifier.padding(top = 16.dp)
        )
    }
}
