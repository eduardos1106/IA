#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <locale.h>
#include <cstddef>

// --- CONSTANTES ---
#define NUM_FEATURES 753       // Total de features no dataset de Parkinson
#define MAX_LINE_SIZE 35000    // Buffer para a linha CSV
#define MAX_ITERATIONS 100    // Número de Weak Learners
#define TRAIN_RATIO 0.80       // 80% para treino
#define FILENAME "parkinsons_disease.csv" // Nome do arquivo a ser carregado

// --- ESTRUTURAS ---
typedef struct {
    int num_samples;
    double** features; // Matriz de dados (linhas x NUM_FEATURES)
    double* labels;    // Vetor de labels (+1.0 ou -1.0)
    int* patient_id;  // <-- ADICIONAR
} Dataset;

typedef struct {
    int feature_index;
    double threshold;
    int direction;     // 1 ou -1
    double alpha;       // Peso de voto
} WeakLearner;

// --- PROTÓTIPOS ---
Dataset* allocate_dataset(int num_samples, int num_features);//le todas as features para uma matriz
void free_dataset(Dataset* data);//libera a memoria alocada da planilha
Dataset* load_csv(const char* filename);//abre e carrega o dataset
void split_and_shuffle(Dataset* full_data, Dataset* train_data, Dataset* test_data);//divide o dataset entre 80% treinamento e 20% de validacao
void initialize_weights(double* weights, int n);//inicia os pesos iniciais para cada decision tree
WeakLearner find_best_stump(Dataset* data, double* weights);//etapa crucial: analisa os pesos de cada decision tree e seleciona qual deve ser alterado
WeakLearner* train_adaboost(Dataset* train_data, int* num_stumps);//faz o teste com as decisions trees
double classify_sample(double* sample_features, WeakLearner* ensemble, int num_stumps);//
void evaluate_model(Dataset* test_data, WeakLearner* ensemble, int num_stumps);//mostra os resultados finais da IA
double get_ensemble_accuracy(Dataset* data, WeakLearner* ensemble, int current_num_stumps);//nao esta definida


// =========================================================================
// IMPLEMENTAÇÕES DE FUNÇÕES AUXILIARES (MEMÓRIA E CARREGAMENTO)
// =========================================================================

// Função para alocar memória para o dataset
Dataset* allocate_dataset(int num_samples, int num_features) {
    if (num_samples <= 0 || num_features <= 0) return NULL;

    Dataset* data = (Dataset*)malloc(sizeof(Dataset));
    if (!data) return NULL;
 
    data->num_samples = num_samples;

    // Alocar array de ponteiros para as linhas
    data->features = (double**)malloc(num_samples * sizeof(double*));
    if (!data->features) { free(data); return NULL; }

    // Alocar os dados em si (num_features por linha)
    for (int i = 0; i < num_samples; i++) {
        data->features[i] = (double*)malloc(num_features * sizeof(double));
        if (!data->features[i]) {
            // Limpeza de erro
            for (int j = 0; j < i; j++) free(data->features[j]);
            free(data->features);
            free(data);
            return NULL;
        }
    }

    // Alocar o vetor de labels
    data->labels = (double*)malloc(num_samples * sizeof(double));
    if (!data->labels) {
        // Limpeza
        for (int i = 0; i < num_samples; i++) free(data->features[i]);
        free(data->features);
        free(data);
        return NULL;
    }

    data->patient_id = (int*)malloc(num_samples * sizeof(int));

 
    return data;
}

// Função para liberar a memória
void free_dataset(Dataset* data) {
    if (data) {
        for (int i = 0; i < data->num_samples; i++) {
            free(data->features[i]);
        }
        free(data->features);
        free(data->labels);
        free(data->patient_id);
        free(data);
    }
}

// Implementação CORRIGIDA do Carregamento do CSV
Dataset* load_csv(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Erro ao abrir arquivo");
        return NULL;
    }

    // 1. Contagem de Linhas
    int total_lines = 0;
    char buffer[MAX_LINE_SIZE];

    // Pula o cabeçalho e conta as linhas
    if (fgets(buffer, sizeof(buffer), file) != NULL) {
        while (fgets(buffer, sizeof(buffer), file) != NULL) {
            total_lines++;
        }
    }

    if (total_lines == 0) {
        printf("AVISO: Nenhuma linha de dados encontrada!\n");
        fclose(file);
        return NULL;
    }

    printf("DEBUG: Encontradas %d linhas de dados\n", total_lines);
    rewind(file);
    fgets(buffer, sizeof(buffer), file); // Pula cabeçalho

    // Alocação do dataset
    Dataset* data = allocate_dataset(total_lines, NUM_FEATURES);
    if (!data) {
        fclose(file);
        return NULL;
    }

    // 2. Leitura e Parsing com verificação
    int i = 0;
    while (fgets(buffer, sizeof(buffer), file) != NULL && i < data->num_samples) {
        char* token;
        int j = 0;
        int actual_columns = 0;

        // Remove quebra de linha do final
        buffer[strcspn(buffer, "\n")] = 0;

        char* temp_line = _strdup(buffer);
        if (!temp_line) {
            printf("ERRO: Falha ao alocar temp_line para linha %d\n", i);
            continue;
        }

        token = strtok(temp_line, ",");

        while (token != NULL) {
            actual_columns++;

            if (j == 0) {
                data->patient_id[i] = atoi(token);
            }
            else if (j >= 1 && j <= NUM_FEATURES) {
                // Features
                if (j - 1 < NUM_FEATURES) {
                    data->features[i][j - 1] = atof(token);
                }
            }
            else if (j == NUM_FEATURES + 1) {
                // Target (última coluna)
                double target_val = atof(token);
                data->labels[i] = (target_val == 1.0) ? 1.0 : -1.0;
            }
            else {
                // Colunas extras - ignorar com aviso
                if (j == NUM_FEATURES + 2) { // Primeira coluna extra
                    printf("AVISO: Linha %d tem mais colunas que o esperado. Esperado: %d, Encontrado: pelo menos %d\n",
                        i + 1, NUM_FEATURES + 2, actual_columns);
                }
            }

            j++;
            token = strtok(NULL, ",");
        }

        // VERIFICAÇÃO CRÍTICA: Número de colunas
        if (actual_columns != NUM_FEATURES + 2) {
            printf("ERRO: Linha %d tem %d colunas, mas esperávamos %d\n",
                i + 1, actual_columns, NUM_FEATURES + 2);
            printf("DEBUG: Conteúdo da linha: %s\n", buffer);
        }

        free(temp_line);
        i++;
    }

    fclose(file);

    // Verificação final
    printf("DEBUG: Carregadas %d amostras de %d esperadas\n", i, total_lines);

    if (i < total_lines) {
        printf("AVISO: Apenas %d de %d amostras foram carregadas\n", i, total_lines);
        // Aqui você poderia realocar o dataset se necessário
    }

    return data;
}

// -------------------------------------------------------------------------
// IMPLEMENTAÇÃO DE FUNÇÕES DE PREPARAÇÃO DE DADOS (EMBARALHAMENTO/SPLIT)
// -------------------------------------------------------------------------


void split_by_patient(Dataset* full, Dataset* train, Dataset* test) {
    int N = full->num_samples;

    // ---- 1. Descobrir todos os IDs únicos ----
    int* unique_ids = (int*)malloc(N * sizeof(int));
    int unique_count = 0;

    for (int i = 0; i < N; i++) {
        int id = full->patient_id[i];
        int found = 0;

        for (int j = 0; j < unique_count; j++) {
            if (unique_ids[j] == id) { found = 1; break; }
        }

        if (!found) {
            unique_ids[unique_count++] = id;
        }
    }

    printf("Total de indivíduos únicos: %d\n", unique_count);

    // ---- 2. Embaralhar IDs únicos ----
    srand(time(NULL));
    for (int i = unique_count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = unique_ids[i];
        unique_ids[i] = unique_ids[j];
        unique_ids[j] = tmp;
    }

    // ---- 3. Calcular quantos IDs vão para treino ----
    int train_ids_count = (int)(unique_count * TRAIN_RATIO);
    int test_ids_count = unique_count - train_ids_count;

    // ---- 4. Criar listas de pertencimento ----
    int* train_ids = (int*)malloc(train_ids_count * sizeof(int));
    int* test_ids = (int*)malloc(test_ids_count * sizeof(int));

    memcpy(train_ids, unique_ids, train_ids_count * sizeof(int));
    memcpy(test_ids, unique_ids + train_ids_count, test_ids_count * sizeof(int));

    // ---- 5. Distribuir as amostras ----
    int t = 0, u = 0;

    for (int i = 0; i < N; i++) {
        int id = full->patient_id[i];

        int in_train = 0;
        for (int j = 0; j < train_ids_count; j++)
            if (train_ids[j] == id) in_train = 1;

        if (in_train) {
            memcpy(train->features[t], full->features[i], NUM_FEATURES * sizeof(double));
            train->labels[t] = full->labels[i];
            train->patient_id[t] = id;
            t++;
        }
        else {
            memcpy(test->features[u], full->features[i], NUM_FEATURES * sizeof(double));
            test->labels[u] = full->labels[i];
            test->patient_id[u] = id;
            u++;
        }
    }

    printf("Treino = %d amostras | Teste = %d amostras\n", t, u);

    free(unique_ids);
    free(train_ids);
    free(test_ids);
}


// -------------------------------------------------------------------------
// IMPLEMENTAÇÕES DO ALGORITMO ADABOOST (CORE)
// -------------------------------------------------------------------------

void initialize_weights(double* weights, int n) {
    for (int i = 0; i < n; i++) {
        weights[i] = 1.0 / n;
    }
}

// O Coração do AdaBoost: Encontra o Decision Stump com menor erro ponderado
WeakLearner find_best_stump(Dataset* data, double* weights) {
    WeakLearner best_stump = {0};
    double min_error = 1.0;

    // Itera sobre todas as FEATURES
    for (int f = 0; f < NUM_FEATURES; f++) {
        // Itera sobre todas as AMOSTRAS para usar seus valores como THRESHOLDS
        for (int i = 0; i < data->num_samples; i++) {
            double current_threshold = data->features[i][f];

            // TESTA AS DUAS DIREÇÕES
            for (int direction = -1; direction <= 1; direction += 2) {
                double current_error = 0.0;

                // Avalia o erro ponderado
                for (int j = 0; j < data->num_samples; j++) {
                    double prediction = 0.0;

                    if (direction == 1) { // h(x) = +1 se feature >= threshold
                        prediction = (data->features[j][f] >= current_threshold) ? 1.0 : -1.0;
                    }
                    else { // h(x) = +1 se feature < threshold
                        prediction = (data->features[j][f] < current_threshold) ? 1.0 : -1.0;
                    }

                    if (prediction != data->labels[j]) {
                        current_error += weights[j];
                    }
                }

                // Atualiza o melhor stump
                if (current_error < min_error) {
                    min_error = current_error;
                    best_stump.feature_index = f;
                    best_stump.threshold = current_threshold;
                    best_stump.direction = direction;
                }
            }
        }
    }
    return best_stump;
}

// Implementação do Treinamento AdaBoost
WeakLearner* train_adaboost(Dataset* train_data, int* num_stumps) {
    WeakLearner* ensemble = (WeakLearner*)malloc(MAX_ITERATIONS * sizeof(WeakLearner));
    double* sample_weights = (double*)malloc(train_data->num_samples * sizeof(double));
    if (!ensemble || !sample_weights) { free(ensemble); return NULL; }

    initialize_weights(sample_weights, train_data->num_samples);
    *num_stumps = 0;

    for (int t = 0; t < MAX_ITERATIONS; t++) {
        WeakLearner current_stump = find_best_stump(train_data, sample_weights);
        double error = 0.0;

        // Re-calcula o erro para ser preciso
        for (int i = 0; i < train_data->num_samples; i++) {
            double h_x = (current_stump.direction == 1)
                ? ((train_data->features[i][current_stump.feature_index] >= current_stump.threshold) ? 1.0 : -1.0)
                : ((train_data->features[i][current_stump.feature_index] < current_stump.threshold) ? 1.0 : -1.0);

            if (h_x != train_data->labels[i]) {
                error += sample_weights[i];
            }
        }

        printf("DEBUG: Iteração %d: Melhor erro encontrado: %.8f\n", t + 1, error);

        // Critério de parada: erro muito baixo (0) ou muito alto (>= 0.5 - chute aleatório)
        if (error < 1e-10 || error >= 0.5 + 1e-8) break;

        // 1. Calcular o peso do classificador (Alpha)
        current_stump.alpha = 0.5 * log((1.0 - error) / error);

        // 2. Atualizar e Normalizar os pesos das amostras
        double sum_weights = 0.0;
        for (int i = 0; i < train_data->num_samples; i++) {
            double h_x = (current_stump.direction == 1) ? ((train_data->features[i][current_stump.feature_index] >= current_stump.threshold) ? 1.0 : -1.0) : ((train_data->features[i][current_stump.feature_index] < current_stump.threshold) ? 1.0 : -1.0);

            // D_i <- D_i * exp(-alpha * y_i * h_i)
            sample_weights[i] *= exp(-current_stump.alpha * train_data->labels[i] * h_x);
            sum_weights += sample_weights[i];
        }

        // Normalização
        for (int i = 0; i < train_data->num_samples; i++) {
            sample_weights[i] /= sum_weights;
        }

        ensemble[*num_stumps] = current_stump;
        (*num_stumps)++;

     
        // Opcional: Mostrar progresso
        printf("Iteração %d: Alpha=%.4f, Erro=%.4f\n", t + 1, current_stump.alpha, error);
    }

    free(sample_weights);
    return ensemble;
}

// -------------------------------------------------------------------------
// IMPLEMENTAÇÕES DE AVALIAÇÃO (CLASSIFY E EVALUATE)
// -------------------------------------------------------------------------

// Retorna erro de previsão (fracão de erros) como double
double erro_previsao(int TP, int TN, int FN, int FP) {
    int total = TP + TN + FP + FN;
    if (total == 0) return 0.0;
    return (double)(FP + FN) / (double)total;
}

double precisao(int TP, int FP) {
    int denom = TP + FP;
    if (denom == 0) return 0.0; // evita divisão por zero
    return (double)TP / (double)denom;
}

double recall(int TP, int FN) {
    int denom = TP + FN;
    if (denom == 0) return 0.0;
    return (double)TP / (double)denom;
}

double f1_score(double precision, double recall) {
    double denom = precision + recall;
    if (denom == 0.0) return 0.0;
    return (2.0 * precision * recall) / denom;
}


// Predição do classificador forte (votação ponderada)
double classify_sample(double* sample_features, WeakLearner* ensemble, int num_stumps) {
    double final_score = 0.0;
    for (int t = 0; t < num_stumps; t++) {
        WeakLearner* stump = &ensemble[t];
        double h_x = 0.0;

        // Calcula a predição do stump atual
        if (stump->direction == 1) {
            h_x = (sample_features[stump->feature_index] >= stump->threshold) ? 1.0 : -1.0;
        }
        else {
            h_x = (sample_features[stump->feature_index] < stump->threshold) ? 1.0 : -1.0;
        }

        // Soma ponderada
        final_score += stump->alpha * h_x;
    }
    // Retorna a classe (+1 ou -1)
    return (final_score >= 0.0) ? 1.0 : -1.0;
}

// Avalia o modelo e imprime a Matriz de Confusão e Acurácia
void evaluate_model(Dataset* test_data, WeakLearner* ensemble, int num_stumps) {
    if (!test_data || test_data->num_samples == 0) {
        printf("Nenhuma amostra de teste para avaliar.\n");
        return;
    }

    int TP = 0, TN = 0, FP = 0, FN = 0;
    int correct_predictions = 0;

    for (int i = 0; i < test_data->num_samples; i++) {
        double predicted = classify_sample(test_data->features[i], ensemble, num_stumps);
        double actual = test_data->labels[i];

        if (predicted == 1.0 && actual == 1.0) {
            TP++;
        }
        else if (predicted == -1.0 && actual == -1.0) {
            TN++;
        }
        else if (predicted == 1.0 && actual == -1.0) {
            FP++;
        }
        else if (predicted == -1.0 && actual == 1.0) {
            FN++;
        }

        if (predicted == actual) {
            correct_predictions++;
        }
    }

    int total_samples = test_data->num_samples;

    // Debug: imprimir matriz de confusão e contagens
    printf("\n--- RESULTADOS DA AVALIAÇÃO (Teste) ---\n");
    printf("\nMatriz de Confusão (Classes: +1 e -1):\n");
    printf("              | Predito +1 (Parkinson) | Predito -1 (Saudável)\n");
    printf("--------------|------------------------|----------------------\n");
    printf("Real +1       | TP: %-18d | FN: %-18d\n", TP, FN);
    printf("Real -1       | FP: %-18d | TN: %-18d\n", FP, TN);
    printf("\nTotal de testes: %d   Acertos: %d\n", total_samples, correct_predictions);

    // Métricas com casts para evitar divisão inteira
    double erro_previ = erro_previsao(TP, TN, FN, FP);
    double precis = precisao(TP, FP);
    double recal = recall(TP, FN);
    double f1score = f1_score(precis, recal);
    double accuracy = (double)(TP + TN) / (double)total_samples;

    printf("\nAcurácia Final: %.4f (Acertos: %d/%d)\n", accuracy, correct_predictions, total_samples);
    printf("Erro de previsão: %.4f\n", erro_previ);
    printf("Precision: %.4f\n", precis);
    printf("Recall: %.4f\n", recal);
    printf("F1 Score: %.4f\n", f1score);
}



// -------------------------------------------------------------------------
// FUNÇÃO PRINCIPAL (MAIN)
// -------------------------------------------------------------------------

int main() {
    setlocale(LC_ALL, "Portuguese");
    printf("--- Projeto AdaBoost para Classificação de Parkinson ---\n");
    printf("Ajustado para %d features e treino/teste 80/20.\n", NUM_FEATURES);

    // 1. Carregamento dos Dados
    Dataset* full_data = load_csv(FILENAME);

    if (full_data) {
        printf("\n--- VERIFICAÇÃO DE DADOS CARREGADOS ---\n");
        printf("Amostra | Target (+1/-1) | Feature 1 (idx 0) | Feature 2 (idx 1) | Feature %d (idx %d)\n", NUM_FEATURES, NUM_FEATURES - 1);

        // Itera sobre as 5 primeiras amostras para inspeção
        int max_check = (full_data->num_samples > 5) ? 5 : full_data->num_samples;

        for (int i = 0; i < max_check; i++) {
            // Imprime o índice da amostra (i) e o label
            printf("%7d | %15.2f |", i, full_data->labels[i]);

            // Imprime as primeiras features
            printf(" %17.5f |", full_data->features[i][0]); // Feature 1
            printf(" %17.5f |", full_data->features[i][1]); // Feature 2

            // Imprime a última feature
            printf(" %17.5f\n", full_data->features[i][NUM_FEATURES - 1]);
        }
        printf("--------------------------------------------------------------------------------------------------------\n");
    }

    if (!full_data) {
        fprintf(stderr, "Falha crítica ao carregar o dataset. Verifique se o arquivo '%s' está na pasta de execução.\n", FILENAME);
        return 1;
    }

    printf("\nDataset carregado com sucesso: %d amostras, %d features.\n", full_data->num_samples, NUM_FEATURES);

    // 2. Alocação e Divisão
    int num_train = (int)(full_data->num_samples * TRAIN_RATIO);
    int num_test = full_data->num_samples - num_train;

    Dataset* train_data = allocate_dataset(num_train, NUM_FEATURES);
    Dataset* test_data = allocate_dataset(num_test, NUM_FEATURES);

    if (!train_data || !test_data) {
        fprintf(stderr, "Erro de alocação de memória para treino/teste.\n");
        free_dataset(full_data);
        if (train_data) free_dataset(train_data);
        if (test_data) free_dataset(test_data);
        return 1;
    }

    // Divisão e Embaralhamento 80/20
    split_by_patient(full_data, train_data, test_data);

    // 3. Treinamento AdaBoost
    int num_stumps = 0;
    printf("\nIniciando treinamento AdaBoost com %d iterações...\n", MAX_ITERATIONS);
    WeakLearner* ensemble = train_adaboost(train_data, &num_stumps);

    if (ensemble && num_stumps > 0) {
        printf("Treinamento concluído. Classificador forte com %d Weak Learners.\n", num_stumps);

        // 4. Avaliação (Teste Final)
        evaluate_model(test_data, ensemble, num_stumps);

        free(ensemble);
    }
    else {
        printf("Aviso: Treinamento interrompido. O modelo não pôde ser construído.\n");
    }

    // 5. Limpeza de memória
    free_dataset(full_data);
    free_dataset(train_data);
    free_dataset(test_data);

    printf("\nExecução finalizada. Limpeza de memória concluída.\n");
    return 0;
}
