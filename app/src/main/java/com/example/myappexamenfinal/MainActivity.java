package com.example.myappexamenfinal;

//Autores José Zambrano y Melanie León

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.myappexamenfinal.ml.ModelLamaria;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;
import java.util.Locale;

public class MainActivity extends AppCompatActivity
        implements TextToSpeech.OnInitListener {

    private static int REQUEST_CAMERA = 111;
    private static int REQUEST_GALLERY = 222;
    private TextToSpeech textToSpeech;
    Bitmap mSelectedImage;
    ImageView mImageView;
    TextView txtResults;

    //Autores José Zambrano y Melanie León
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = findViewById(R.id.imgCapturada);
        txtResults = findViewById(R.id.twRespuesta);

        // Inicializa TextToSpeech
        textToSpeech = new TextToSpeech(this, this);
    }
    public void abrirGaleria (View view){
        Intent i = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(i, REQUEST_GALLERY);
    }
    public void abrirCamara(View view) {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE_SECURE);
        startActivityForResult(cameraIntent, REQUEST_CAMERA);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            handleImageCapture(requestCode, data);
        }
    }
    private void handleImageCapture(int requestCode, Intent data) {
        try {
            mSelectedImage = requestCode == REQUEST_CAMERA ? (Bitmap) data.getExtras().get("data")
                    : MediaStore.Images.Media.getBitmap(getContentResolver(), data.getData());
            displayCapturedImage();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void displayCapturedImage() {
        mImageView.setImageBitmap(mSelectedImage);
    }

    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            // Configura el idioma del TextToSpeech, por ejemplo, en español
            int result = textToSpeech.setLanguage(new Locale("es", "EC"));

            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS", "El idioma no está disponible.");
            } else {
                // Tu instancia de TextToSpeech está lista para usar
            }
        } else {
            Log.e("TTS", "La inicialización falló.");
        }
    }

    private void leerTexto(String texto) {
        textToSpeech.speak(texto, TextToSpeech.QUEUE_FLUSH, null, null);
    }
    public void reconocerLugar(View view){
        try {
            //Definir Estiquetas de acuerdo a su archivo "labels.txt" generado por la Plataforma de creación del Modelo
            String[] etiquetas = {"Baños","Biblioteca","Centro Médico","Comedor","Facultad de Ciencias Agrarias y Forestales"
            ,"Facultad de Ciencias de la Industria y Producción","Facultad de Ciencias de Ingeniería","Laboratorio de Acuicultura",
            "Laboratorio Industrial","Laboratorio de Suelos","Parqueadero","Polideportivo"};

            ModelLamaria model = ModelLamaria.newInstance(getApplicationContext());
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(convertirImagenATensorBuffer(mSelectedImage));

            ModelLamaria.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            txtResults.setText(obtenerEtiquetayProbabilidad(etiquetas, outputFeature0.getFloatArray()));

            leerTexto(obtenerEtiquetayProbabilidad(etiquetas, outputFeature0.getFloatArray()));

            model.close();
        } catch (Exception e) {
            txtResults.setText(e.getMessage());
        }
    }
    public ByteBuffer convertirImagenATensorBuffer(Bitmap mSelectedImage){

        Bitmap imagen = Bitmap.createScaledBitmap(mSelectedImage, 224, 224, true);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[224 * 224];
        imagen.getPixels(intValues, 0, imagen.getWidth(), 0, 0, imagen.getWidth(), imagen.getHeight());

        int pixel = 0;

        for(int i = 0; i <  imagen.getHeight(); i ++){
            for(int j = 0; j < imagen.getWidth(); j++){
                int val = intValues[pixel++]; // RGB
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
            }
        }
        return  byteBuffer;
    }

    public String obtenerEtiquetayProbabilidad(String[] etiquetas, float[] probabilidades){

        float valorMayor=Float.MIN_VALUE;
        int pos=-1;
        for (int i = 0; i < probabilidades.length; i++) {
            if (probabilidades[i] > valorMayor) {
                valorMayor = probabilidades[i];
                pos = i;
            }
        }
        return "Predicción: " + etiquetas[pos] + ", Probabilidad: " + (new DecimalFormat("0.00").format(probabilidades[pos] * 100)) + "%";
    }
}