plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.example.xnndemo"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.example.xnndemo"
        minSdk = 29
        targetSdk = 34
        ndk { abiFilters += listOf("x86_64", "arm64-v8a") }
        versionCode = 1
        versionName = "1.0"
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // Kotlin JVM target for Android plugin
    kotlinOptions {
        jvmTarget = "17"
    }
}
// Kotlin toolchain (helps Gradle pick the right JDK for Kotlin)
kotlin {
    jvmToolchain(17)
}
dependencies {
    implementation("org.pytorch:executorch-android:0.7.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    // (No need to add kotlin-stdlib explicitly; the plugin handles it)
    // Unit tests (run on JVM)
    testImplementation("junit:junit:4.13.2")

// Instrumented tests (run on device/emulator) — optional but common
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")

}
