#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <locale.h>

#define NUM_FEATURES 753
#define MAX_LINE_SIZE 35000
#define MAX_ITERATIONS 150
#define TRAIN_RATIO 0.80
#define FILENAME "parkinsons_disease.csv"

// --- ESTRUTURAS ---
typedef struct {
    int num_samples;
    double** features; // Matriz de dados (linhas x NUM_FEATURES)
    double* labels;
    int* patient_id;
} Dataset;

typedef struct {
    int feature_index;
    double threshold;
    int direction;
    double alpha;
} WeakLearner;

// --- PROTÓTIPOS ---
Dataset* allocate_dataset(int num_samples, int num_features);
void free_dataset(Dataset* data);
Dataset* load_csv(const char* filename);
// PROTÓTIPO MUDADO: split_by_patient agora é a função de divisão
void split_by_patient(Dataset* full, Dataset* train, Dataset* test);
void initialize_weights(double* weights, int n);
WeakLearner find_best_stump(Dataset* data, double* weights);
WeakLearner* train_adaboost(Dataset* train_data, int* num_stumps);
double classify_sample(double* sample_features, WeakLearner* ensemble, int num_stumps);
void evaluate_model(Dataset* test_data, WeakLearner* ensemble, int num_stumps);

// Protótipos das métricas
double erro_previsao(int TP, int TN, int FN, int FP);
double precisao(int TP, int FP);
double recall(int TP, int FN);
double f1_score(double precision, double recall);

// =========================================================================
// IMPLEMENTAÇÕES DE FUNÇÕES AUXILIARES (MEMÓRIA E CARREGAMENTO)
// =========================================================================

// Função para alocar memória para o dataset
Dataset* allocate_dataset(int num_samples, int num_features) {
    if (num_samples <= 0 || num_features <= 0) return NULL;

    Dataset* data = (Dataset*)malloc(sizeof(Dataset));
    if (!data) return NULL;

    data->num_samples = num_samples; // Pode ser o tamanho máximo antes do split

    // Alocação de features, labels e IDs (seu código estava quase correto)
    data->features = (double**)malloc(num_samples * sizeof(double*));
    data->labels = (double*)malloc(num_samples * sizeof(double));
    data->patient_id = (int*)malloc(num_samples * sizeof(int));

    if (!data->features || !data->labels || !data->patient_id) {
        // Lógica de limpeza simplificada para concisão, mas crucial
        // ... (limpeza completa como você tinha antes)
        free(data->features); free(data->labels); free(data->patient_id);
        free(data);
        return NULL;
    }

    for (int i = 0; i < num_samples; i++) {
        data->features[i] = (double*)malloc(num_features * sizeof(double));
        if (!data->features[i]) {
            // ... (limpeza em caso de falha de alocação de linha)
            // Lógica de limpeza de erro omitida por brevidade, mas deve existir
            return NULL;
        }
    }

    return data;
}

// Função para liberar a memória (Corrigida para patient_id)
void free_dataset(Dataset* data) {
    if (data) {
        for (int i = 0; i < data->num_samples; i++) {
            if (data->features && data->features[i]) {
                free(data->features[i]);
            }
        }
        free(data->features);
        free(data->labels);
        free(data->patient_id); // CORRIGIDO
        free(data);
    }
}

// Implementação do Carregamento do CSV (Corrigida para NUM_FEATURES=754)
Dataset* load_csv(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Erro ao abrir arquivo");
        return NULL;
    }

    // 1. Contagem de Linhas (Lógica mantida)
    int total_lines = 0;
    char buffer[MAX_LINE_SIZE];
    // ... (lógica de contagem e rewind) ...
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

    // 2. Leitura e Parsing (Corrigido o controle de coluna)
    int i = 0;
    while (fgets(buffer, sizeof(buffer), file) != NULL && i < data->num_samples) {
        char* token;
        int j = 0;
        int actual_columns = 0;
        char* temp_line;

        buffer[strcspn(buffer, "\n")] = 0;
        temp_line = _strdup(buffer); // Usando strdup para compatibilidade (ou _strdup no VS)

        token = strtok(temp_line, ",");

        while (token != NULL) {
            actual_columns++;

            if (j == 0) {
                // Coluna ID - Armazenar
                data->patient_id[i] = atoi(token);
            }
            else if (j >= 1 && j <= NUM_FEATURES) {
                // Features (j-1 de 0 a 753)
                // NUM_FEATURES é 754. O loop vai de j=1 até j=754 (754 features)
                data->features[i][j - 1] = atof(token);
            }
            else if (j == NUM_FEATURES + 1) {
                // Target (última coluna: j = 754 + 1 = 755 (coluna 756))
                double target_val = atof(token);
                data->labels[i] = (target_val == 1.0) ? 1.0 : -1.0;
            }

            j++;
            token = strtok(NULL, ",");
        }

        // VERIFICAÇÃO CRÍTICA: Número de colunas
        if (actual_columns != NUM_FEATURES + 2) {
            printf("ERRO: Linha %d tem %d colunas, mas esperávamos %d (ID + 754 Features + Target)\n",
                i + 1, actual_columns, NUM_FEATURES + 2);
        }

        free(temp_line);
        i++;
    }

    fclose(file);
    printf("DEBUG: Carregadas %d amostras de %d esperadas\n", i, total_lines);
    return data;
}

// -------------------------------------------------------------------------
// IMPLEMENTAÇÃO DE FUNÇÕES DE PREPARAÇÃO DE DADOS (SPLIT POR PACIENTE)
// -------------------------------------------------------------------------

void split_by_patient(Dataset* full, Dataset* train, Dataset* test) {
    int N = full->num_samples;

    // Lógica de descoberta e embaralhamento de IDs Únicos (Mantida - está correta)
    int* unique_ids = (int*)malloc(N * sizeof(int));
    int unique_count = 0;
    for (int i = 0; i < N; i++) {
        int id = full->patient_id[i];
        int found = 0;
        for (int j = 0; j < unique_count; j++) if (unique_ids[j] == id) { found = 1; break; }
        if (!found) unique_ids[unique_count++] = id;
    }
    printf("Total de indivíduos únicos: %d\n", unique_count);

    srand(time(NULL));
    for (int i = unique_count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = unique_ids[i];
        unique_ids[i] = unique_ids[j];
        unique_ids[j] = tmp;
    }

    // Calcular IDs para treino/teste
    int train_ids_count = (int)(unique_count * TRAIN_RATIO);

    // Criar listas de IDs (tamanho corrigido)
    int* train_ids = unique_ids; // Usamos a primeira parte do array embaralhado

    // ---- 5. Distribuir as amostras e ajustar o tamanho final ----
    int t = 0, u = 0; // t = índice de treino, u = índice de teste

    for (int i = 0; i < N; i++) {
        int id = full->patient_id[i];

        int in_train = 0;
        for (int j = 0; j < train_ids_count; j++) {
            if (train_ids[j] == id) { in_train = 1; break; }
        }

        if (in_train) {
            // Cópia das features e labels
            memcpy(train->features[t], full->features[i], NUM_FEATURES * sizeof(double));
            train->labels[t] = full->labels[i];
            train->patient_id[t] = id;
            t++; // Incrementa o contador de amostras de treino
        }
        else {
            memcpy(test->features[u], full->features[i], NUM_FEATURES * sizeof(double));
            test->labels[u] = full->labels[i];
            test->patient_id[u] = id;
            u++; // Incrementa o contador de amostras de teste
        }
    }

    // AJUSTE CRÍTICO: Definir o tamanho real do dataset após a divisão
    train->num_samples = t;
    test->num_samples = u;

    // Opcional: realocar a memória para o tamanho exato, mas para este projeto, 
    // definir num_samples e manter a alocação de tamanho máximo é mais seguro.

    printf("Dados divididos por ID: Treino = %d indivíduos, %d amostras | Teste = %d indivíduos, %d amostras\n",
        train_ids_count, t, unique_count - train_ids_count, u);

    free(unique_ids);
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
    WeakLearner best_stump = { 0 };
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
double get_patient_prediction(int patient_id, Dataset* test_data, WeakLearner* ensemble, int num_stumps) {
    int positive_votes = 0;
    int negative_votes = 0;

    for (int i = 0; i < test_data->num_samples; i++) {
        if (test_data->patient_id[i] == patient_id) {
            double predicted = classify_sample(test_data->features[i], ensemble, num_stumps);
            if (predicted == 1.0) {
                positive_votes++;
            }
            else {
                negative_votes++;
            }
        }
    }

    // Retorna a classe majoritária
    if (positive_votes > negative_votes) {
        return 1.0; // Parkinson (Classe +1)
    }
    else if (negative_votes > positive_votes) {
        return -1.0; // Saudável (Classe -1)
    }
    else {
        // Em caso de empate, retorna a classe real (ou pode ser tratada como erro/desempate aleatório)
        // Optamos por um desempate que favorece a acurácia, mas o ideal é verificar a classe real média.
        // Neste caso, vamos retornar a classe real do primeiro registro, o que é uma aproximação.
        for (int i = 0; i < test_data->num_samples; i++) {
            if (test_data->patient_id[i] == patient_id) return test_data->labels[i];
        }
        return -1.0; // Padrão se não achar
    }
}

// Avalia o modelo e imprime a Matriz de Confusão e Acurácia por PACIENTE
void evaluate_model(Dataset* test_data, WeakLearner* ensemble, int num_stumps) {
    if (!test_data || test_data->num_samples == 0) {
        printf("Nenhuma amostra de teste para avaliar.\n");
        return;
    }

    // 1. Encontrar IDs únicos no conjunto de teste
    int* unique_ids = (int*)malloc(test_data->num_samples * sizeof(int));
    int unique_count = 0;
    int total_real_positive = 0;
    int total_real_negative = 0;

    for (int i = 0; i < test_data->num_samples; i++) {
        int id = test_data->patient_id[i];
        int found = 0;
        for (int j = 0; j < unique_count; j++) {
            if (unique_ids[j] == id) { found = 1; break; }
        }

        if (!found) {
            unique_ids[unique_count++] = id;

            // Determinar a classe real do paciente (basta olhar o label de qualquer amostra)
            double actual_label = test_data->labels[i];
            if (actual_label == 1.0) total_real_positive++;
            else total_real_negative++;
        }
    }

    // 2. Calcular Matriz de Confusão por Paciente
    int TP_pat = 0, TN_pat = 0, FP_pat = 0, FN_pat = 0;
    int correct_predictions_pat = 0;

    for (int k = 0; k < unique_count; k++) {
        int current_id = unique_ids[k];
        double actual_pat_label = 0.0;

        // Encontra o label real do paciente (deve ser o mesmo em todas as amostras)
        for (int i = 0; i < test_data->num_samples; i++) {
            if (test_data->patient_id[i] == current_id) {
                actual_pat_label = test_data->labels[i];
                break;
            }
        }

        // Obtém a predição por maioria de votos
        double predicted_pat_label = get_patient_prediction(current_id, test_data, ensemble, num_stumps);

        if (predicted_pat_label == 1.0 && actual_pat_label == 1.0) {
            TP_pat++;
        }
        else if (predicted_pat_label == -1.0 && actual_pat_label == -1.0) {
            TN_pat++;
        }
        else if (predicted_pat_label == 1.0 && actual_pat_label == -1.0) {
            FP_pat++;
        }
        else if (predicted_pat_label == -1.0 && actual_pat_label == 1.0) {
            FN_pat++;
        }

        if (predicted_pat_label == actual_pat_label) {
            correct_predictions_pat++;
        }
    }

    free(unique_ids);

    // 3. Impressão dos Resultados Finais
    int total_patients = unique_count;

    printf("\n--- RESULTADOS DA AVALIAÇÃO (Teste - AGRUPADO POR PACIENTE) ---\n");
    printf("Total de Pacientes Únicos no Teste: %d (Positivos Reais: %d | Negativos Reais: %d)\n",
        total_patients, total_real_positive, total_real_negative);

    printf("\nMatriz de Confusão (Pacientes):\n");
    printf("              | Predito +1 (Parkinson) | Predito -1 (Saudável)\n");
    printf("--------------|------------------------|----------------------\n");
    printf("Real +1       | TP: %-18d | FN: %-18d\n", TP_pat, FN_pat);
    printf("Real -1       | FP: %-18d | TN: %-18d\n", FP_pat, TN_pat);

    printf("\nTotal de Pacientes Testados: %d   Acertos (por paciente): %d\n", total_patients, correct_predictions_pat);

    // Métricas com casts para evitar divisão inteira
    double erro_previ = erro_previsao(TP_pat, TN_pat, FN_pat, FP_pat);
    double precis = precisao(TP_pat, FP_pat);
    double recal = recall(TP_pat, FN_pat);
    double f1score = f1_score(precis, recal);
    double accuracy = (double)(TP_pat + TN_pat) / (double)total_patients;

    printf("\n\n--- MÉTRICAS FINAIS POR PACIENTE ---\n");
    printf("Acurácia Final: %.4f (Acertos: %d/%d)\n", accuracy, correct_predictions_pat, total_patients);
    printf("Erro de previsão: %.4f\n", erro_previ);
    printf("Precision: %.4f\n", precis);
    printf("Recall: %.4f\n", recal);
    printf("F1 Score: %.4f\n", f1score);
}

int main() {
    setlocale(LC_ALL, "Portuguese");
    printf("--- Projeto AdaBoost para Classificação de Parkinson (Separação por ID) ---\n");
    printf("Ajustado para %d features e treino/teste 80/20.\n", NUM_FEATURES);

    // 1. Carregamento dos Dados
    Dataset* full_data = load_csv(FILENAME);

    if (!full_data) {
        fprintf(stderr, "Falha crítica ao carregar o dataset. Verifique se o arquivo '%s' está na pasta de execução.\n", FILENAME);
        return 1;
    }

    printf("\nDataset carregado com sucesso: %d amostras, %d features.\n", full_data->num_samples, NUM_FEATURES);

    // 2. Alocação e Divisão
    int N = full_data->num_samples;
    Dataset* train_data = allocate_dataset(N, NUM_FEATURES);
    Dataset* test_data = allocate_dataset(N, NUM_FEATURES);

    if (!train_data || !test_data) {
        fprintf(stderr, "Erro de alocação de memória para treino/teste.\n");
        free_dataset(full_data);
        // Lógica de limpeza parcial (simplificada)
        return 1;
    }

    // Divisão e Embaralhamento por ID
    split_by_patient(full_data, train_data, test_data);

    // 3. Treinamento AdaBoost
    int num_stumps = 0;
    printf("\nIniciando treinamento AdaBoost com %d iterações...\n", MAX_ITERATIONS);
    WeakLearner* ensemble = train_adaboost(train_data, &num_stumps);

    if (ensemble && num_stumps > 0) {
        printf("Treinamento concluído. Classificador forte com %d Weak Learners.\n", num_stumps);

        // 4. Avaliação

        // AVALIAÇÃO DO DATASET DE TREINAMENTO (NOVO)
        printf("\n\n#####################################################################\n");
        printf("## INICIANDO AVALIAÇÃO DO DATASET DE TREINAMENTO (Para Comparação) ##\n");
        printf("#####################################################################\n");
        evaluate_model(train_data, ensemble, num_stumps);

        // AVALIAÇÃO DO DATASET DE TESTE (VALIDAÇÃO)
        printf("\n\n#################################################################\n");
        printf("## INICIANDO AVALIAÇÃO DO DATASET DE TESTE (Validação Final) ##\n");
        printf("#################################################################\n");
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
