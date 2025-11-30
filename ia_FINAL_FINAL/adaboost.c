#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <locale.h>

// constantes de numero
#define BUFFER_LINHAS 35000
#define ITERACOES 180
#define TAXA_APRENDIZADO 0.2

// constantes que possuem os nomes dos arquivos
#define DATASET_ORIGINAL "pd_speech_features.csv"
#define ARQ_TREINO "Treino.csv"
#define ARQ_VALIDACAO "Teste.csv"

typedef struct {//struct dos datasets pra montar eles
    int num_amostras;
    int num_features;      
    double** features;
    double* rotulos;
    int* id_paciente;
} Dataset;

typedef struct {//struct do adaboost
    int indice_caracteristica;
    double limiar;
    int direcao;
    double alpha;
} WeakLearner;

//prototipos da maioria da funções (eu acho)
void preparar_arquivos_treino_teste();
Dataset* aloca_dataset(int num_amostras, int num_features);
void free_dataset(Dataset* data);
Dataset* carrega_arquivo(const char* filename);
void inicia_pesos(double* pesos, int n);
WeakLearner define_melhor_stump(Dataset* data, double* pesos);
WeakLearner* treina_algoritmo_adaboost(Dataset* train_data, int* num_stumps);
double classifica_amostra(double* sample_features, WeakLearner* ensemble, int num_stumps);
double avalia_paciente(int patient_id, Dataset* test_data, WeakLearner* ensemble, int num_stumps);
void resultados_modelo(Dataset* test_data, WeakLearner* ensemble, int num_stumps, const char* title);

// protótipos das métricas
double erro_previsao(int TP, int TN, int FN, int FP);
double precisao(int TP, int FP);
double recall(int TP, int FN);
double f1_score(double precision, double recall);

// =========================================================================
// função de pré-processamento dos dados pra ver se melhora o algoritmo
void normaliza_dataset(Dataset* train_data, Dataset* test_data) {
    if (!train_data || !test_data || train_data->num_features != test_data->num_features) {
        printf("ERRO: Dados invalidos para normalizacao.\n");
        return;
    }

    int N = train_data->num_features;
    double* min_vals = (double*)malloc(N * sizeof(double)); // minimo de cada feature
    double* max_vals = (double*)malloc(N * sizeof(double)); // maximo de cada feature

    if (!min_vals || !max_vals) {
        printf("ERRO DE MEMORIA: Nao foi possivel alocar min/max.\n");
        return;
    }

   
    for (int f = 0; f < N; f++) {//encontra o valor minimo e maximo de cada feature no cojunto de train_set
        min_vals[f] = train_data->features[0][f];
        max_vals[f] = train_data->features[0][f];

        for (int i = 1; i < train_data->num_amostras; i++) {
            double val = train_data->features[i][f];
            if (val < min_vals[f]) min_vals[f] = val;
            if (val > max_vals[f]) max_vals[f] = val;
        }
    }

    //normalização das features, evitando que pequenas diferenças cause um grande estrago
    for (int f = 0; f < N; f++) {
        double range = max_vals[f] - min_vals[f];
        if (range < 1e-10) continue; // evita que o numero seja zero e de pau

        for (int i = 0; i < train_data->num_amostras; i++) {
            train_data->features[i][f] = (train_data->features[i][f] - min_vals[f]) / range;//normaliza a feature, subtraindo o valor dela pelo minimo, e dividindo pela diferença de maximo e minimo
        }
    }

    for (int f = 0; f < N; f++) {//aplica a mesma coisa para o conjunto de test_set
        double range = max_vals[f] - min_vals[f];
        if (range < 1e-10) continue;

        for (int i = 0; i < test_data->num_amostras; i++) {
            test_data->features[i][f] = (test_data->features[i][f] - min_vals[f]) / range;
        }
    }

    printf("Normalizacao Min-Max aplicada com sucesso. (Range: 0 a 1)\n");

    free(min_vals);
    free(max_vals);
}

// =========================================================================
// parte que vai implementar a separação de arquivos e a divisao 80 pra 20
void preparar_arquivos_treino_teste() {
    printf("\n--- 1. Processando Arquivo Original e Separando Dados POR PACIENTE (ID) ---\n");
    FILE* file = fopen(DATASET_ORIGINAL, "r");
    if (!file) {
        printf("ERRO CRITICO: Nao foi possivel abrir '%s'.\n", DATASET_ORIGINAL);
        exit(1);
    }


    char buffer[BUFFER_LINHAS];
    char* header = NULL;
    if (fgets(buffer, BUFFER_LINHAS, file) != NULL) {
        header = _strdup(buffer);//le o cabeçalho que possui informação nao essencial
    }
    else {
        fclose(file);
        printf("Arquivo vazio ou sem cabeçalho.\n");
        exit(1);
    }


    char** all_lines = NULL;//variaveis para ler as linhas
    int line_count = 0;
    int capacity = 0;

    int* all_ids = NULL;//variaveis para ler os ids
    int id_capacity = 0;

    while (fgets(buffer, BUFFER_LINHAS, file)) {

        if (line_count >= capacity) {
            if (capacity == 0) {
                capacity = 512;
            }
            else {
                capacity = capacity * 2;
            }

            all_lines = (char**)realloc(all_lines, capacity * sizeof(char*));
            all_ids = (int*)realloc(all_ids, capacity * sizeof(int));//vai alocar a memoria de acordo com a quantidade de linhas lida

            if (!all_lines || !all_ids) {//linha pra avisar se deu ruim

                exit(1);
            }
        }
        all_lines[line_count] = _strdup(buffer);

        // Extrair ID do paciente (assumindo que é o primeiro token)
        char* temp_line = _strdup(buffer);
        char* token = strtok(temp_line, ",");
        if (token != NULL) {
            all_ids[line_count] = atoi(token);//utilizando token (que separa strings), a primeira coisa é sempre pegar o id
        }
        else {
            all_ids[line_count] = -1; // ID inválido
        }
        free(temp_line);

        line_count++;
    }
    fclose(file);

    if (line_count == 0) {
        free(header);
        // ... (tratamento de arquivo vazio)
        exit(1);
    }

    // identifica ids unicos - vai ler tudo de volta e ver se tem algum repetido, sao esses que vao ser embaralhados
    int* unique_ids = (int*)malloc(line_count * sizeof(int));
    int unique_count = 0;
    for (int i = 0; i < line_count; i++) {
        int id = all_ids[i];
        int found = 0;
        for (int j = 0; j < unique_count; j++) {
            if (unique_ids[j] == id) { found = 1; break; }
        }
        if (!found) {
            unique_ids[unique_count++] = id;
        }
    }
    free(all_ids);

    printf("Total de registros (linhas/amostras) encontrados: %d\n", line_count);
    printf("Total de pacientes unicos encontrados: %d\n", unique_count);

    //embaralhar e dividir os IDs Únicos
    printf("Embaralhando IDs dos pacientes...\n");

    srand((unsigned int)time(NULL));//embaralha utilizando um algoritmo de troca com valores de j aleatorio entre 1 e i
    for (int i = unique_count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = unique_ids[i];
        unique_ids[i] = unique_ids[j];
        unique_ids[j] = temp;
    }

    //divide 80/20 a quantidade de pacientes pra treino e validação
    int train_id_size = (int)floor(unique_count * 0.8);

    int* is_train = (int*)calloc(unique_count, sizeof(int));

    for (int i = 0; i < train_id_size; i++) {//cria uma flag pra saber se o id é de treino
        is_train[i] = 1; // 1 significa Treino
    }

    // 5. Escrever os arquivos de Treino e Teste
    FILE* f_train = fopen(ARQ_TREINO, "w");
    FILE* f_test = fopen(ARQ_VALIDACAO, "w");

    if (!f_train || !f_test) {
        printf("Erro ao criar arquivos de saída.\n");
        exit(1);
    }

    fprintf(f_train, "%s", header);
    fprintf(f_test, "%s", header);

    int train_lines_count = 0;
    int test_lines_count = 0;

    for (int i = 0; i < line_count; i++) {
        
        char* temp_line = _strdup(all_lines[i]);//cria variaveis temporarias pra poder ler a linha
        char* token = strtok(temp_line, ",");

        int current_id;
        
        if (token != NULL) {
            current_id = atoi(token);//extrai o id da linha atual, transforma utilizando o atoi
        }
        else {
            current_id = -1;
        }
  

        free(temp_line);

        // verifica o grupo do ID
        for (int k = 0; k < unique_count; k++) {
            if (unique_ids[k] == current_id) {
                if (k < train_id_size) { // pertence ao grupo de Treino (índices 0 a train_id_size-1)
                    fprintf(f_train, "%s", all_lines[i]);//vai escrevendo as linhas
                    train_lines_count++;
                }
                else { // pertence ao grupo de Teste
                    fprintf(f_test, "%s", all_lines[i]);
                    test_lines_count++;
                }
                break;
            }
        }
    }

    fclose(f_train);
    fclose(f_test);

    printf("Arquivos gerados por Paciente ID:\n");//finalmente mostra como foi separado os dados
    printf("  '%s': %d pacientes | %d amostras\n", ARQ_TREINO, train_id_size, train_lines_count);
    printf("  '%s': %d pacientes | %d amostras\n", ARQ_VALIDACAO, unique_count - train_id_size, test_lines_count);

    // Liberação de memória
    free(header);
    free(unique_ids);
    free(is_train);
    for (int j = 0; j < line_count; j++) {
        free(all_lines[j]);
    }
    free(all_lines);
}

// =========================================================================
//  parte de memoria e alocação para os arquivos
Dataset* aloca_dataset(int num_amostras, int num_features) {//aloca a memoria do dataset que vai ser criado (treino ou validação)
    if (num_amostras <= 0 || num_features <= 0) return NULL;//se o numero de amostras for negativo (absurdo) retorna NULL

    Dataset* data = (Dataset*)malloc(sizeof(Dataset));
    if (!data) return NULL;//se deu ruim, avisa

    data->num_amostras = num_amostras;
    data->num_features = num_features;
    data->features = (double**)malloc(num_amostras * sizeof(double*));//aloca a quantidade de cada coisa com o numero de amostras (80% ou 20% do dataset)
    data->rotulos = (double*)malloc(num_amostras * sizeof(double));
    data->id_paciente = (int*)malloc(num_amostras * sizeof(int));

    if (!data->features || !data->rotulos || !data->id_paciente) {//se alguma coisa deu ruim, libera a memoria e retorna NULL
        free(data->features); free(data->rotulos); free(data->id_paciente); free(data);
        return NULL;
    }

    for (int i = 0; i < num_amostras; i++) {
        data->features[i] = (double*)malloc(num_features * sizeof(double));//cada amostra vai guardar a quantidade de features
        if (!data->features[i]) {
            // liberar tudo e retornar NULL
            for (int k = 0; k < i; k++) free(data->features[k]);
            free(data->features); free(data->rotulos); free(data->id_paciente); free(data);
            return NULL;
        }
    }
    return data;
}

void free_dataset(Dataset* data) {//funcao que libera a memoria de todos os elementos da struct dataset (um de treinamento e outro de validaççao)
    if (data) {
        for (int i = 0; i < data->num_amostras; i++) {
            if (data->features && data->features[i]) free(data->features[i]);//da free em cada feature
        }
        free(data->features);
        free(data->rotulos);
        free(data->id_paciente);
        free(data);
    }
}

static int count_tokens_in_line(const char* line) {
    // conta vírgulas + 1
    int cnt = 1;
    for (const char* p = line; *p; ++p) if (*p == ',') ++cnt;
    return cnt;
}

Dataset* carrega_arquivo(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Erro ao abrir arquivo: %s\n", filename);
        return NULL;
    }

    char buffer[BUFFER_LINHAS];

    // ler cabeçalho
    if (!fgets(buffer, sizeof(buffer), file)) { fclose(file); return NULL; }

    
    int total_lines = 0;// contar linhas restantes e determinar o número de features
    long pos_after_header = ftell(file);
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        total_lines++;
    }

    if (total_lines == 0) { 
        fclose(file); 
        return NULL; 
    }

    // Voltar para linha após cabeçalho
    fseek(file, pos_after_header, SEEK_SET);

  
    if (!fgets(buffer, sizeof(buffer), file)) { fclose(file); return NULL; }
    buffer[strcspn(buffer, "\n")] = 0;
    int token_count = count_tokens_in_line(buffer); // número total de colunas

  
    int num_features = token_count - 2;// assumindo: Primeira coluna é ID, Última coluna é Rótulo

    if (num_features <= 0) {
        fclose(file);
        printf("Formato de linha invalido em %s. Esperado (ID, Features..., Rótulo).\n", filename);
        return NULL;
    }

    
    fseek(file, pos_after_header, SEEK_SET);//volta pro inicio

    // 3. Alocar e ler o Dataset
    Dataset* data = aloca_dataset(total_lines, num_features);
    if (!data) { 
        fclose(file); 
        return NULL; 
    }

    data->num_features = num_features;

    int i = 0;
    while (fgets(buffer, sizeof(buffer), file) != NULL && i < data->num_amostras) {
        buffer[strcspn(buffer, "\n")] = 0;
        char* temp_line = _strdup(buffer);
        char* token = strtok(temp_line, ",");

        
        if (token == NULL) { free(temp_line); break; }
        data->id_paciente[i] = atoi(token);//le o id, primeiro token sempre

        token = strtok(NULL, ",");

        int feature_idx = 0;
        while (token != NULL && feature_idx < data->num_features) {
            data->features[i][feature_idx] = atof(token);
            feature_idx++;//tudo o resto vai ser feature
            token = strtok(NULL, ",");
        }
        
        if (token != NULL) {
            double target_val = atof(token);
            // rotula como 1.0 (Parkinson) ou -1.0 (Saudável)
            data->rotulos[i] = (target_val == 1.0) ? 1.0 : -1.0;
        }
        free(temp_line);
        i++;
    }
    fclose(file);
    return data;
}

// =========================================================================
//  implementação do adaboost

void inicia_pesos(double* pesos, int n) {
    for (int i = 0; i < n; i++) {
        pesos[i] = 1.0 / n;//inicia os pesos de cada amostra para (1/numero de amostras) - pois a soma deve ser = 1
    }
}

WeakLearner define_melhor_stump(Dataset* data, double* pesos) {/*acha a regra de classificação mais simples que comete o menor erro ponderado -
    é uma pequena arvore de decisão que vai ser juntada no ensemble*/
    WeakLearner melhor_stump = { 0 };//guarda melhor regra de classificao aqui
    double erro_minimo = 1.0;//erro minimo começa com 1 (o maximo)

    for (int f = 0; f < data->num_features; f++) {//itera sobre cada caracteristicas (features)
        for (int i = 0; i < data->num_amostras; i++) {//itera sobre cada amostra do dataset
            double limiar_atual = data->features[i][f];//aqui o limiar é o valor da feature f da amostra i que vai ser testado (um "limite")
            for (int direcao = -1; direcao <= 1; direcao += 2) {//teste dois possiveis casos, direcao = -1 ou direcao = 1
                double erro_atual = 0.0;//define o erro para 0 pois vai começar a somar ele
                for (int j = 0; j < data->num_amostras; j++) {//itera sobre todas as amostras j para calcular o erro - através dele detecta se é um bom stum
                    double predicao; //define a predicao do stump (-1 ou 1) para a amostra j
                    if (direcao == 1) {
                        if (data->features[j][f] >= limiar_atual) predicao = 1.0;
                        else predicao = -1.0;
                    }
                    else {
                        if (data->features[j][f] < limiar_atual) predicao = 1.0;
                        else predicao = -1.0;
                    }

                    if (predicao != data->rotulos[j]) {
                        erro_atual += pesos[j]; //se o stump errou, aumenta o erro dele pelo peso da feature -> DAR MAIS FOCO A ESSA FEATURE
                    }
                }
                if (erro_atual < erro_minimo) {//define se o stump é bom: se o erro atual é menor que o erro minimo
                    erro_minimo = erro_atual;
                    melhor_stump.indice_caracteristica = f;//salva cada caracteristica do stump
                    melhor_stump.limiar = limiar_atual;
                    melhor_stump.direcao = direcao;
                }
            }
        }
    }
    return melhor_stump;//após iterar sobre todas as caracteristicas, retorna o melhor stump que da mais foco a feature necessaria
}

WeakLearner* treina_algoritmo_adaboost(Dataset* train_data, int* num_stumps) {//aqui ele junta os fraquinho com os forte pra fazer o megazord
    WeakLearner* ensemble = (WeakLearner*)malloc(ITERACOES * sizeof(WeakLearner));//aloca a memoria de acordo com a quantidade de iteracoes do megazord
    double* sample_weights = (double*)malloc(train_data->num_amostras * sizeof(double));//aloca a memoria para o vetor que guarda os pesos
    inicia_pesos(sample_weights, train_data->num_amostras); // Passar os rótulos//inicia os pesos iniciais distribuidos pela quantidade de amostras
    *num_stumps = 0;//quantidade de stumps inicial é igual a 0 ->nenhuma arvore de decisao foi criada ainda

    for (int t = 0; t < ITERACOES; t++) {//vai iterando pela quantidade maxima de iterações (stumps)
        WeakLearner current_stump = define_melhor_stump(train_data, sample_weights);//acha a melhor arvore de decisao utilizando os pesos respectivos
        double error = 0.0;//inicializa o erro ponderado =0 (vai ser calculado depois para medir a eficacia do stump)

        for (int i = 0; i < train_data->num_amostras; i++) {
            double h_x; //calcula a predicao (h_x) de acordo com o limiar do melhor stump
            if (current_stump.direcao == 1) {
                if (train_data->features[i][current_stump.indice_caracteristica] >= current_stump.limiar) h_x = 1.0;
                else h_x = -1.0;
            }
            else {
                if (train_data->features[i][current_stump.indice_caracteristica] < current_stump.limiar) h_x = 1.0;
                else h_x = -1.0;
            }

            if (h_x != train_data->rotulos[i]) {//se errou, aumenta o peso com o peso da amostra
                error += sample_weights[i];
            }

        }

        if (error < 1e-10 || error >= 0.5) {
            break;//se for um erro absurdo de ruim, fecha o codigo (ele ta chutando se tem parkinson ou nao)
        }

        current_stump.alpha = 0.5 * log((1.0 - error) / error);//calcula o alfa: metrica que diz o quao bom é aquele stump de fato pra depois saber qual a parte mais forte do megazord
        // quanto menor o erro, maior o alfa -> mais confiavel pois errou pouco

        printf("(Alfa sem taxa: %.4f)", current_stump.alpha);

        current_stump.alpha = TAXA_APRENDIZADO * 0.5 * log((1.0 - error) / error);

        double sum_weights = 0.0;//define a soma dos pesos para começar a normalização (qual peso vai aumentar e qual vai diminuir)

        for (int i = 0; i < train_data->num_amostras; i++) {//itera sobre todas as amostras
            double h_x; //recalcula a mesma predicao feita antes
            if (current_stump.direcao == 1) {
                if (train_data->features[i][current_stump.indice_caracteristica] >= current_stump.limiar) h_x = 1.0;
                else h_x = -1.0;
            }
            else {
                if (train_data->features[i][current_stump.indice_caracteristica] < current_stump.limiar) h_x = 1.0;
                else h_x = -1.0;
            }

            sample_weights[i] *= exp(-current_stump.alpha * train_data->rotulos[i] * h_x);//calculo do peso da amostra - o que vai determinar se o voto dele tem mais peso ou nao
            sum_weights += sample_weights[i];
        }

        for (int i = 0; i < train_data->num_amostras; i++) {
            sample_weights[i] /= sum_weights;//normalização dos pesos (soma=1)
        }

        ensemble[*num_stumps] = current_stump;//adiciona o stump pra um pedaço do megazord
        (*num_stumps)++;//aumenta o numero de stumps que o ensemble vai ter
        printf("\nIteracao %d: Alpha=%.4f, Erro=%.4f\n", t + 1, current_stump.alpha, error);//imprime o resultado para ver como estao indo os stumps
    }
    free(sample_weights);//libera a memoria alocada inicialmente
    return ensemble;//retorna o megazord com os classificadores fracos
}

// =========================================================================
// 4. parte de avalição e resultados
double erro_previsao(int TP, int TN, int FN, int FP) {
    int total = TP + TN + FP + FN;
    if (total == 0) {
        return 0.0;
    }
    else {
        return (double)(FP + FN) / total;
    }//formual do erro
}
double precisao(int TP, int FP) {
    if (TP + FP == 0) {
        return 0.0;
    }
    else {
        return (double)TP / (TP + FP);
    }//formula da precisao
}
double recall(int TP, int FN) {
    if (TP + FN == 0) {
        return 0.0;
    }
    else {
        return (double)TP / (TP + FN);
    }//formula do recall
}
double f1_score(double precision, double recall) {
    if (precision + recall == 0) {
        return 0.0;
    }
    else {
        return (2.0 * precision * recall) / (precision + recall);
    }//formula do f1score
}

double classifica_amostra(double* sample_features, WeakLearner* ensemble, int num_stumps) {//aqui ele classifica a amostra (feature)
    double final_score = 0.0;
    for (int t = 0; t < num_stumps; t++) {//itera sobre todos os stumps criados
        double h_x;
        if (ensemble[t].direcao == 1) {
            if (sample_features[ensemble[t].indice_caracteristica] >= ensemble[t].limiar) h_x = 1.0;
            else h_x = -1.0;
        }
        else {
            if (sample_features[ensemble[t].indice_caracteristica] < ensemble[t].limiar) h_x = 1.0;
            else h_x = -1.0;
        }
        final_score += ensemble[t].alpha * h_x;//vai somando a quantidade de votos de cada stump SOBRE A FEATURE EM DESTAQUE
    }

    if (final_score >= 0.0) return 1.0;
    else return -1.0;
}

double avalia_paciente(int patient_id, Dataset* test_data, WeakLearner* ensemble, int num_stumps) {//aqui ele junta as amostras pra avaliar o paciente
    int positive_votes = 0;
    int negative_votes = 0;
    for (int i = 0; i < test_data->num_amostras; i++) {
        if (test_data->id_paciente[i] == patient_id) {
            if (classifica_amostra(test_data->features[i], ensemble, num_stumps) == 1.0) positive_votes++;
            else negative_votes++;//cada feature vai ganhando um voto de acordo com o final score
        }
    }
    if (positive_votes > negative_votes) return 1.0;//se tiver mais votos positivos -> nao tem parkinson
    if (negative_votes > positive_votes) return -1.0;//se tiver mais votos negativos -> tem parkinson

    for (int i = 0; i < test_data->num_amostras; i++) {
        if (test_data->id_paciente[i] == patient_id) return test_data->rotulos[i];//se tiver um empate, retorna o rotulo inicial do paciente
    }
    return -1.0;
}

void resultados_modelo(Dataset* test_data, WeakLearner* ensemble, int num_stumps, const char* title) {//aqui mostra os resultados
    if (!test_data || test_data->num_amostras == 0) {
        printf("\n--- %s ---\n", title);
        printf("Nenhuma amostra para avaliar.\n");
        return;
    }

    int* unique_ids = (int*)malloc(test_data->num_amostras * sizeof(int));//aloca memoria para identificar os ids dos pacientes e assim avaliar por paciente
    int unique_count = 0;//contador de pacientes

    for (int i = 0; i < test_data->num_amostras; i++) {
        int id = test_data->id_paciente[i];//obtem o id do paciente
        int found = 0;//flag pra saber se o id ja foi lido

        for (int j = 0; j < unique_count; j++) {
            if (unique_ids[j] == id)
            {
                found = 1; break;
            }//checa se o id ja esta na lista
        }

        if (!found) {
            unique_ids[unique_count++] = id;//se nao, adiciona no vetor que possui todos os ids
        }
    }

    int TP = 0, TN = 0, FP = 0, FN = 0;//cria as variaveis que vao ser passadas para as metricas
    int correct = 0;

    for (int k = 0; k < unique_count; k++) {//aqui ele começa a validação real
        int ID = unique_ids[k];//recebe o id do paciente
        double actual = 0.0;//variavel que guarda o real estado do paciente (1 ou -1)

        // Busca label real
        for (int i = 0; i < test_data->num_amostras; i++) {
            if (test_data->id_paciente[i] == ID) { actual = test_data->rotulos[i]; break; }//o atual vai recebe o estado
        }

        double predicted = avalia_paciente(ID, test_data, ensemble, num_stumps);//chama a funcao pra avaliar o paciente, e guarda o veredito em predicted

        if (predicted == 1.0 && actual == 1.0) TP++;//ifs que vai definir se é um TP, TN....
        else if (predicted == -1.0 && actual == -1.0) TN++;
        else if (predicted == 1.0 && actual == -1.0) FP++;
        else if (predicted == -1.0 && actual == 1.0) FN++;

        if (predicted == actual) correct++;//correct é o numero de acertos
    }
    free(unique_ids);//libera memoria do vetor que guardava os ids

    printf("\n--- %s ---\n", title);//matriz de confusao
    printf("Matriz de Confusao (Pacientes - Total: %d)", unique_count);
    if (unique_count > 100) {
        printf("(80%):\n");
    }
    else {
        printf("(20%):\n");
    }
    printf("Real +1 (Parkinson) | TP: %d | FN: %d\n", TP, FN);
    printf("Real -1 (Saudavel)  | FP: %d | TN: %d\n", FP, TN);

    double acc = (double)correct / unique_count;
    double prec = precisao(TP, FP);
    double rec = recall(TP, FN);
    double f1 = f1_score(prec, rec);

    printf("Acuracia: %.4f | Error: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f\n", acc,1-acc, prec, rec, f1);//mostra todos as metricas correspondentes
}

// MAIN
int main() {
    setlocale(LC_ALL, "Portuguese");
    printf("=== PROJETO MACHINE LEARNING: ADABOOST.=== \n Eduardo Scalcon e Pedro Pilato\n");

    preparar_arquivos_treino_teste();//gera os dois arquivos separados para treino e validação

    printf("\n--- 2. Carregando Treino (%s) ---\n", ARQ_TREINO);
    Dataset* train_data = carrega_arquivo(ARQ_TREINO);//carrega e lê o arquivo de treino
    if (!train_data) { 
        printf("Falha ao carregar treino.\n"); 
        return 1; 
    }

    printf("--- 3. Carregando Teste (%s) ---\n", ARQ_VALIDACAO);
    Dataset* test_data = carrega_arquivo(ARQ_VALIDACAO);//carrega e lê o arquivo de validação
    if (!test_data) { 
        printf("Falha ao carregar teste.\n"); 
        return 1; 
    }

    printf("\n--- 3.5. Pré-processamento de Dados ---\n");
    normaliza_dataset(train_data, test_data); //aqui ele trata os dados antes de mandar treinar, pois tem numeros muito afastados, causando ruido no algoritmo que deixa ele paia

    int num_stumps = 0;
    printf("\n--- 4. Iniciando Treinamento AdaBoost ---\n");//começa de fato o treinamento do adaboost
    WeakLearner* ensemble = treina_algoritmo_adaboost(train_data, &num_stumps);

    if (ensemble && num_stumps > 0) {

        resultados_modelo(train_data, ensemble, num_stumps, "5.1. Avaliacao do TREINAMENTO");//avaliação do dataset de treinamento

        printf("\n---------------------------------------------------------------------------\n");

        resultados_modelo(test_data, ensemble, num_stumps, "5.2. Avaliacao do TESTE (VALIDACAO)");//avaliação do dataset de validação (o que importa)

        free(ensemble);
    }
    else {
        printf("Erro no treinamento.\n");//se tem algum valor essencial errado, avisa
    }

    free_dataset(train_data);//libera a memoria das variaveis utilizados para treinamentos e validacao
    free_dataset(test_data);

    printf("\nProcesso Finalizado.\n");
    return 0;
}
