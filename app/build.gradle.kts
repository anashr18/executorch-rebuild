plugins { id("com.android.application") }

android {
  namespace = "com.example.etdemo"
  compileSdk = 34

  defaultConfig {
    applicationId = "com.example.etdemo"
    minSdk = 29
    targetSdk = 34
    // Support both emulator and real device
    ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    versionCode = 1
    versionName = "1.0"
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(
        getDefaultProguardFile("proguard-android-optimize.txt"),
        "proguard-rules.pro"
      )
    }
  }
}

dependencies {
  // ExecuTorch AAR from Maven Central
  implementation("org.pytorch:executorch-android:0.7.0")
}
