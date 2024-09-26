#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <raylib.h>

#define ERROR_EXIT(...) { fprintf(stderr, "Error in file %s at line %d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); exit(1); }

#define BSWAP(n) ((((n) & 0x000000FF) << 24) | (((n) & 0x0000FF00) << 8) | (((n) & 0x00FF0000) >> 8) | (((n) & 0xFF000000) >> 24))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define IMAGE_SIZE 28
#define HIDDEN_SIZE 256
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001f
#define BATCH_SIZE 64
#define TRAINING_SPLIT 0.8f

#define IMG_PATH "data/images"
#define LABEL_PATH "data/labels"

typedef unsigned char byte;

typedef struct layer {
    float* weights; // represents a matrix
    float* biases;
    int inputSize, outputSize;
} Layer;

typedef struct network {
    Layer hidden, output;
} Network;

typedef struct inputData {
    byte* images;
    byte* labels;
    int count;
} InputData;

// W ~ U(-sqrt(6)/in, sqrt(6)/in)
Layer LayerInit(int inputSize, int outputSize) {
    const int n = inputSize * outputSize;
    const float scale = sqrtf(2.f / inputSize);

    Layer layer = {
        .inputSize = inputSize,
        .outputSize = outputSize,
        .weights = malloc(n * sizeof(float)),
        .biases = calloc(outputSize, sizeof(float))
    };

    if (!layer.weights || !layer.biases) {
        ERROR_EXIT("Failed to allocate for weights or biases\n");
    }

    for (int i = 0; i < n; i++) {
        layer.weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.f * scale;
    }

    return layer;
}

// move the data through each layer of the network and
// apply linear transformations and activation functions
void Forward(const Layer* layer, const float* input, float* output) {
    for (int i = 0; i < layer->outputSize; i++) {
        output[i] = layer->biases[i];
        for (int j = 0; j < layer->inputSize; j++) {
            output[i] += input[j] * layer->weights[j * layer->outputSize + i];
        }
    }
}

// updates weights and biases based on the gradients
// propagate the error from the output layer until reaching the
// input layer, passing through the higgen layer(s)
void Backward(const Layer* layer, const float* input,
              const float* outputGrad, float* inputGrad, float lr) {
    for (int i = 0; i < layer->outputSize; i++) {
        for (int j = 0; j < layer->inputSize; j++) {
            const int ind = j * layer->outputSize + i;

            // new weight = old weight - learning rate * gradient of loss
            //                                           with respect to weight
            layer->weights[ind] -= lr * outputGrad[i] * input[j];

            // gradient of loss with respect to input j is the sum of
            // gradient of loss with respect to each output i times the weight
            // connecting input j to output i over all outputs
            if (inputGrad) {
                inputGrad[j] += outputGrad[i] * layer->weights[ind];
            }
        }

        // new bias = old bias - learning rate * gradient of loss
        //                                       with respect to loss
        layer->biases[i] -= lr * outputGrad[i];
    }
}

// e^z_i / sum{j=1, K}(e^z_j)
void Softmax(float* input, int size) {
    float max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    float sum = 0.f;
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// one training iteration
void Train(const Network* net, const float* input, int label, float lr) {
    float hiddenOutput[HIDDEN_SIZE], finalOutput[OUTPUT_SIZE];
    float hiddenGrad[HIDDEN_SIZE] = {0}, outputGrad[OUTPUT_SIZE] = {0};

// forward pass
    // input to hidden layer
    Forward(&net->hidden, input, hiddenOutput);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hiddenOutput[i] = MAX(hiddenOutput[i], 0.f);// ReLU
    }

    // hidden to output layer
    Forward(&net->output, hiddenOutput, finalOutput);
    Softmax(finalOutput, OUTPUT_SIZE);

// compute output gradient
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        outputGrad[i] = finalOutput[i] - (float)(i == label);
    }

// backward pass
    // output layer to hidden layer
    Backward(&net->output, hiddenOutput, outputGrad, hiddenGrad, lr);

    // backpropagate through ReLU
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hiddenGrad[i] *= hiddenOutput[i] > 0 ? 1 : 0; // ReLU derivative
    }

    // hidden layer to input layer
    Backward(&net->hidden, input, hiddenGrad, NULL, lr);
}

// implements the forward pass for running inference
// instead of computing gradients it returns the most likely digit prediction
int Predict(const Network* net, const float* input) {
    float hiddenOutput[HIDDEN_SIZE], finalOutput[OUTPUT_SIZE];

    Forward(&net->hidden, input, hiddenOutput);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hiddenOutput[i] = MAX(hiddenOutput[i], 0.f);
    }

    Forward(&net->output, hiddenOutput, finalOutput);
    Softmax(finalOutput, OUTPUT_SIZE);

    int maxInd = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (finalOutput[i] > finalOutput[maxInd]) {
            maxInd = i;
        }
    }

    return maxInd;
}

void ShuffleData(const InputData* data) {
    for (int i = data->count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        for (int k = 0; k < INPUT_SIZE; k++) {
            const int i0 = i * INPUT_SIZE + k;
            const int i1 = j * INPUT_SIZE + k;
            const byte temp = data->images[i0];
            data->images[i0] = data->images[i1];
            data->images[i1] = temp;
        }
        const byte temp = data->labels[i];
        data->labels[i] = data->labels[j];
        data->labels[j] = temp;
    }
}

void ReadImages(const char* path, byte** images, int* imageCount);
void ReadLabels(const char* path, byte** labels, int* labelCount);

int main(int argc, char** argv) {
    Network net;
    InputData data = {0};
    float img[INPUT_SIZE];

    srand(time(NULL));
    net.hidden = LayerInit(INPUT_SIZE, HIDDEN_SIZE);
    net.output = LayerInit(HIDDEN_SIZE, OUTPUT_SIZE);

    ReadImages(IMG_PATH, &data.images, &data.count);
    ReadLabels(LABEL_PATH, &data.labels, &data.count);

    ShuffleData(&data);

    const int trainSize = (int)(data.count * TRAINING_SPLIT);
    const int testSize = data.count - trainSize;

    const char* statusMessage = "";

    InitWindow(1280, 720, "MNIST");

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);

        const Vector2 mp = GetMousePosition();

        static int epochs = 0;
        static int totalEpochs = 0;
        for (int epoch = 0; epoch < epochs; epoch++, totalEpochs++) {
            float totalLoss = 0.f;
            for (int i = 0; i < trainSize; i += BATCH_SIZE) {
                for (int j = 0; j < BATCH_SIZE && i + j < trainSize; j++) {
                    const int ind = i + j;
                    for (int k = 0; k < INPUT_SIZE; k++) {
                        img[k] = data.images[ind * INPUT_SIZE + k] / 255.f;
                    }

                    Train(&net, img, data.labels[ind], LEARNING_RATE);

                    float hiddenOutput[HIDDEN_SIZE], finalOutput[OUTPUT_SIZE];
                    Forward(&net.hidden, img, hiddenOutput);
                    for (int k = 0; k < HIDDEN_SIZE; k++) {
                        hiddenOutput[k] = MAX(hiddenOutput[k], 0.f);
                    }
                    Forward(&net.output, hiddenOutput, finalOutput);
                    Softmax(finalOutput, OUTPUT_SIZE);

                    totalLoss += -logf(finalOutput[data.labels[ind]] + 1e-10f);
                }
            }

            int correct = 0;
            for (int i = trainSize; i < data.count; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    img[j] = data.images[i * INPUT_SIZE + j] / 255.f;
                }
                if (Predict(&net, img) == data.labels[i]) {
                    correct++;
                }
            }

            char msg[64];
            sprintf(msg, "Epoch %d, Acc %.3f, Avg Loss %.5f", totalEpochs + 1, (float)correct / testSize * 100, totalLoss / trainSize);
            statusMessage = msg;
            puts(msg);
        }
        epochs = 0;

// region ui
        const float fontSize = 32;
        const float fontBorder = 3.f;
        const Color buttonCol = {38, 38, 38, 255};
        const Color hoverCol = {50, 50, 50, 255};

        const char* loadText = "Load Network";
        const int loadWidth = MeasureText(loadText, fontSize);
        const Rectangle loadRect = { 600 - fontBorder, 50 - fontBorder, loadWidth + fontBorder * 4.f, fontSize + fontBorder * 2.f};
        Color loadCol = buttonCol;
        if (mp.x >= loadRect.x && mp.x < loadRect.x + loadRect.width && mp.y >= loadRect.y && mp.y < loadRect.y + loadRect.height) {
            loadCol = hoverCol;
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                FILE* saveData = fopen("save/save.mnistSave", "rb");
                if (!saveData) {
                    ERROR_EXIT("Failed to open save data\n");
                }

                fread(net.hidden.weights, sizeof(float), INPUT_SIZE * HIDDEN_SIZE, saveData);
                fread(net.hidden.biases, sizeof(float), HIDDEN_SIZE, saveData);
                fread(&net.hidden.inputSize, sizeof(int), 1, saveData);
                fread(&net.hidden.outputSize, sizeof(int), 1, saveData);
                fread(net.output.weights, sizeof(float), HIDDEN_SIZE * OUTPUT_SIZE, saveData);
                fread(net.output.biases, sizeof(float), OUTPUT_SIZE, saveData);
                fread(&net.output.inputSize, sizeof(int), 1, saveData);
                fread(&net.output.outputSize, sizeof(int), 1, saveData);

                statusMessage = "Loaded network";
            }
        }
        DrawRectangle(loadRect.x, loadRect.y, loadRect.width, loadRect.height, loadCol);
        DrawText(loadText, loadRect.x + fontBorder * 2.f, loadRect.y + fontBorder, fontSize, WHITE);

        const char* saveText = "Save Network";
        const int saveWidth = MeasureText(saveText, fontSize);
        const Rectangle saveRect = { loadRect.x + loadRect.width + 100, 50 - fontBorder, saveWidth + fontBorder * 4.f, fontSize + fontBorder * 2.f};
        Color saveCol = buttonCol;
        if (mp.x >= saveRect.x && mp.x < saveRect.x + saveRect.width && mp.y >= saveRect.y && mp.y < saveRect.y + saveRect.height) {
            saveCol = hoverCol;
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                FILE* saveData = fopen("save/save.mnistSave", "wb");
                if (!saveData) {
                    ERROR_EXIT("Failed to open file\n");
                }

                fwrite(net.hidden.weights, sizeof(float), INPUT_SIZE * HIDDEN_SIZE, saveData);
                fwrite(net.hidden.biases, sizeof(float), HIDDEN_SIZE, saveData);
                fwrite(&net.hidden.inputSize, sizeof(int), 1, saveData);
                fwrite(&net.hidden.outputSize, sizeof(int), 1, saveData);
                fwrite(net.output.weights, sizeof(float), HIDDEN_SIZE * OUTPUT_SIZE, saveData);
                fwrite(net.output.biases, sizeof(float), OUTPUT_SIZE, saveData);
                fwrite(&net.output.inputSize, sizeof(int), 1, saveData);
                fwrite(&net.output.outputSize, sizeof(int), 1, saveData);

                statusMessage = "Saved network";
            }
        }
        DrawRectangle(saveRect.x, saveRect.y, saveRect.width, saveRect.height, saveCol);
        DrawText(saveText, saveRect.x + fontBorder, saveRect.y + fontBorder, fontSize, WHITE);

        const char* addText = "Add 5 Epochs";
        const int addWidth = MeasureText(addText, fontSize);
        const Rectangle addRect = { loadRect.x, loadRect.y + fontSize + 50, addWidth + fontBorder * 4.f, fontSize + fontBorder * 2.f};
        Color addCol = buttonCol;
        if (mp.x >= addRect.x && mp.x < addRect.x + addRect.width && mp.y >= addRect.y && mp.y < addRect.y + addRect.height) {
            addCol = hoverCol;
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                epochs += 5;
                statusMessage = "Added epochs, training...";
            }
        }
        DrawRectangle(addRect.x, addRect.y, addRect.width, addRect.height, addCol);
        DrawText(addText, addRect.x + fontBorder, addRect.y + fontBorder, fontSize, WHITE);

        static float drawnImg[INPUT_SIZE];
        const char* resetText = "Reset Drawing";
        const int resetWidth = MeasureText(resetText, fontSize);
        const Rectangle resetRect = { loadRect.x, addRect.y + fontSize + 50, resetWidth + fontBorder * 4.f, fontSize + fontBorder * 2.f};
        Color resetCol = buttonCol;
        if (mp.x >= resetRect.x && mp.x < resetRect.x + resetRect.width && mp.y >= resetRect.y && mp.y < resetRect.y + resetRect.height) {
            resetCol = hoverCol;
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                memset(drawnImg, 0, INPUT_SIZE * sizeof(float));
            }
        }
        DrawRectangle(resetRect.x, resetRect.y, resetRect.width, resetRect.height, resetCol);
        DrawText(resetText, resetRect.x + fontBorder, resetRect.y + fontBorder, fontSize, WHITE);

        const int drawPosX = 10;
        const int drawPosY = 50;
        const float drawScale = 20.f;
        const float brushRad = 1.5f;
        for (int y = 0, i = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++, i++) {
                const float xp = drawPosX + x * drawScale;
                const float yp = drawPosY + y * drawScale;
                const Color col = (Color){drawnImg[i] * 255, drawnImg[i] * 255, drawnImg[i] * 255, 255};
                DrawRectangle(xp, yp, drawScale, drawScale, col);

                if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                    continue;
                }

                const float dx = mp.x - (xp + drawScale / 2);
                const float dy = mp.y - (yp + drawScale / 2);
                const float dist = sqrtf(dx * dx + dy * dy);

                if (dist >= drawScale * brushRad) {
                    continue;
                }

                const float factor = 1.0f - (dist / (drawScale * brushRad));
                drawnImg[i] = fminf(drawnImg[i] + factor * 0.1f, 1.0f);
            }
        }
        DrawRectangleLines(drawPosX, drawPosY, drawScale * 28, drawScale * 28, WHITE);

        char buf[32];
        sprintf(buf, "Predicted: %d", Predict(&net, drawnImg));
        DrawText(buf, 10, 10, 36, WHITE);

        DrawText(statusMessage, 10, GetScreenHeight() - fontSize - 10, fontSize, WHITE);
// end region ui

        EndDrawing();
    }

    free(net.hidden.weights);
    free(net.hidden.biases);
    free(net.output.weights);
    free(net.output.biases);
    free(data.images);
    free(data.labels);
}

void ReadImages(const char* path, byte** images, int* imageCount) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        ERROR_EXIT("Failed to open file at %s\n", path);
    }

    int temp;
    fread(&temp, sizeof(int), 1, file);

    fread(imageCount, sizeof(int), 1, file);
    *imageCount = BSWAP(*imageCount);

    int rows, cols;
    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);

    *images = malloc((*imageCount) * IMAGE_SIZE * IMAGE_SIZE);
    fread(*images, sizeof(byte), (*imageCount) * IMAGE_SIZE * IMAGE_SIZE, file);

    fclose(file);
}

void ReadLabels(const char* path, byte** labels, int* labelCount) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        ERROR_EXIT("Failed to open file at: %s\n", path);
    }

    int temp;
    fread(&temp, sizeof(int), 1, file);

    fread(labelCount, sizeof(int), 1, file);
    *labelCount = BSWAP(*labelCount);

    *labels = malloc(*labelCount);
    fread(*labels, sizeof(byte), *labelCount, file);

    fclose(file);
}
