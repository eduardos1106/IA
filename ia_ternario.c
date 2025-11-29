#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <locale.h>

// constantes de numero
#define NUM_FEATURES 753
#define BUFFER_LINHAS 35000
#define ITERACOES 108

// constantes que possuem os nomes dos arquivos
#define DATASET_ORIGINAL "pd_speech_features.csv"
#define ARQ_TREINO "Treino.csv"
#define ARQ_VALIDACAO "Teste.csv"


typedef struct {//struct dos datasets pra montar eles
    int num_amostras;
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

// Protótipos das métricas
double erro_previsao(int TP, int TN, int FN, int FP);
double precisao(int TP, int FP);
double recall(int TP, int FN);
double f1_score(double precision, double recall);

// =========================================================================
// 1. IMPLEMENTAÇÃO DA SEPARAÇÃO DE ARQUIVOS
// =========================================================================

void troca_linha(char** a, char** b) {//funcao que troca de linha, utilizada para embaralhamento
    char* temp = *a;
    *a = *b;
    *b = temp;
}

void embaralha(char** array, int n) {//aqui vai trocar as linhas do array de acordo com o numero de elementos
    srand((unsigned int)time(NULL));//gera valores aleatorios para trocar de linha, o unsigned serve pra evitar valores negativos
    for (int i = n - 1; i > 0; i--) {//varre o vetor de tras para frente
        int j = rand() % (i + 1);//gera um indice entre 0 e i
        troca_linha(&array[i], &array[j]);//troca as linhas de i com j
    }
}

void preparar_arquivos_treino_teste() {
    printf("\n--- 1. Processando Arquivo Original e Separando Dados ---\n");
    FILE* file = fopen(DATASET_ORIGINAL, "r");
    if (!file) {
        printf("ERRO CRITICO: Nao foi possivel abrir '%s'.\n", DATASET_ORIGINAL);
        exit(1);
    }

    char buffer[BUFFER_LINHAS];


    char* header_lixo = fgets(buffer, BUFFER_LINHAS, file);//aqui ele pula o cabeçalho, que possui informação desnecessaria
    if (!header_lixo) {
        fclose(file);
        return; //se deu ruim, avisa
    }


    char* header_real = _strdup(buffer);//cria uma copia do cabeçalho da segunda linha pra garantir que nao tenha o lixo anterior
    if (!fgets(buffer, BUFFER_LINHAS, file)) {
        strcpy(header_real, buffer);
    }
    else {
        free(header_real);
        header_real = _strdup(buffer);
    }

    // Contar dados
    long data_pos = ftell(file);//ftell passa o ponteiro que marca onde os dados estao sendo lidos agora no arquivo, que é aonde queremos começar
    int count = 0;
    while (fgets(buffer, BUFFER_LINHAS, file)){
        count++;
}//conta todas as linhas do arquivo

    printf("Total de registros encontrados: %d\n", count);//mostra a quantidade de linhas
    if (count == 0) { fclose(file); exit(1); }//se nao contou nenhuma linha, da erro

   
    fseek(file, data_pos, SEEK_SET);//volta a leitura pro inicio do arquivo, que é o queriamos (aqui evita ter que pular aquele cabeçalho com lixo)
    char** lines = (char**)malloc(count * sizeof(char*));//aloca a quantidade de linhas com o que foi lido

    int i = 0;
    while (fgets(buffer, BUFFER_LINHAS, file) && i < count) {
        lines[i] = _strdup(buffer);//le e armazena cada linha de dados (_stdup é uma funcao que duplica a string para outra, mais facil de utilizar que strcpy)
        i++;
    }
    fclose(file);//fecha o arquivo, pois ja temos todas as linhas que queriamos

  
    printf("Embaralhando dados...\n");
    embaralha(lines, count);//embaralha as linhas extraidas do dataset original

    //divide 80/20
    int train_size = (int)(count * 0.8);
    int test_size = count - train_size;

   
    FILE* f_train = fopen(ARQ_TREINO, "w");//abre no modo escrita
    fprintf(f_train, "%s", header_real);//escreve o cabeçalho que queremos e salvamos antes
    for (int j = 0; j < train_size; j++) fprintf(f_train, "%s", lines[j]);//formata o arquivo treino com todos os pacientes e features respectivamente
    fclose(f_train);

    
    FILE* f_test = fopen(ARQ_VALIDACAO, "w");//abre no modo escrita
    fprintf(f_test, "%s", header_real);//salva cabeçalho
    for (int j = train_size; j < count; j++) fprintf(f_test, "%s", lines[j]);//formata o arquvio de validação com os pacientes e features
    fclose(f_test);

    printf("Arquivos gerados: '%s' (%d) e '%s' (%d).\n", ARQ_TREINO, train_size, ARQ_VALIDACAO, test_size);//mostra os arquivos gerados juntamente com a quantidade de features cada uma

   
    free(header_real);//da free nas variaveis alocadas dinamicamente na funcao
    for (int j = 0; j < count; j++) {
        free(lines[j]);
    }
    free(lines);
}

// =========================================================================
// 2. IMPLEMENTAÇÕES DE MEMÓRIA E CARREGAMENTO
// =========================================================================

Dataset* aloca_dataset(int num_amostras, int num_features) {//aloca a memoria do dataset que vai ser criado (treino ou validação)
    if (num_amostras <= 0 || num_features <= 0) return NULL;//se o numero de amostras for negativo (absurdo) retorna NULL

    Dataset* data = (Dataset*)malloc(sizeof(Dataset));
    if (!data) return NULL;//se deu ruim, avisa

    data->num_amostras = num_amostras;
    data->features = (double**)malloc(num_amostras * sizeof(double*));//aloca a quantidade de cada coisa com o numero de amostras (80% ou 20% do dataset)
    data->rotulos = (double*)malloc(num_amostras * sizeof(double));
    data->id_paciente = (int*)malloc(num_amostras * sizeof(int));

    if (!data->features || !data->rotulos || !data->id_paciente) {//se alguma coisa deu ruim, libera a memoria e retorna NULL
        free(data); 
        return NULL;
    }

    for (int i = 0; i < num_amostras; i++) {
        data->features[i] = (double*)malloc(num_features * sizeof(double));//cada amostra vai guardar a quantidade de features
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

Dataset* carrega_arquivo(const char* filename) {
    FILE* file = fopen(filename, "r");//abre o arquivo para leitura
    if (!file) {
        printf("Erro ao abrir arquivo: %s\n", filename);
        return NULL;
    }

    int total_lines = 0;//variavel que guarda a quantidade de linhas
    char buffer[BUFFER_LINHAS];

    
    fgets(buffer, sizeof(buffer), file);//simplesmente pula o cabeçalho que contem informação desnecessaria

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        total_lines++;//conta a quantidade de linhas que o dataset possui
    }

    if (total_lines == 0) {
        fclose(file);//se nao contou nada, retornar NULL
        return NULL;
    }

    rewind(file);//volta pro inicio do arquivo
    fgets(buffer, sizeof(buffer), file); // pula o cabeçalho com informação desnecessaria

    Dataset* data = aloca_dataset(total_lines, NUM_FEATURES);//aloca o dataset com a quantidade de linhas lida anteriormente
    if (!data) { fclose(file); return NULL; }//se deu ruim, avisa

    int i = 0;
    while (fgets(buffer, sizeof(buffer), file) != NULL && i < data->num_amostras) {//le linha por linha
        buffer[strcspn(buffer, "\n")] = 0;
        char* temp_line = _strdup(buffer);// _strdup duplica uma string alocada dinamicamente e passa como ponteiro para outro char
        char* token = strtok(temp_line, ",");//como o arquivo é csv, separa por virgula

        int j = 0;
        while (token != NULL) {//iteracoes que vao indo palavra por palavra por causa da funcao token da biblioteca string.h
            if (j == 0) { 
                data->id_paciente[i] = atoi(token);//id do paciente - passado como string (atoi)
            }
            else if (j >= 1 && j <= NUM_FEATURES) { 
                data->features[i][j - 1] = atof(token);//passa as features (!!) como um numero (atof)
            }
            else if (j == NUM_FEATURES + 1) { // 
                double target_val = atof(token);
                // SUBSTITUIÇÃO DO TERNÁRIO: (target_val == 1.0) ? 1.0 : -1.0
                if (target_val == 1.0) {
                    data->rotulos[i] = 1.0;
                } else {
                    data->rotulos[i] = -1.0;
                }//passa o rotulo (-1 ou +1) pra depois validar e ver se errou ou acertou
            }
            j++;
            token = strtok(NULL, ",");//pula para a proxima palavra tendo como limite de separação a virgula
        }
        free(temp_line);//libera a memoria da variavel temporaria para separar cada string
        i++;
    }
    fclose(file);
    return data;
}

// =========================================================================
// 3. IMPLEMENTAÇÕES DO ALGORITMO ADABOOST
// =========================================================================

void inicia_pesos(double* pesos, int n) {
    for (int i = 0; i < n; i++) {
        pesos[i] = 1.0 / n;//inicia os pesos de cada amostra para (1/numero de amostras) - pois a soma deve ser = 1
    }
}

WeakLearner define_melhor_stump(Dataset* data, double* pesos) {/*acha a regra de classificação mais simples que comete o menor erro ponderado -
    é uma pequena arvore de decisão que vai ser juntada no ensemble*/
    WeakLearner melhor_stump = { 0 };//guarda melhor regra de classificao aqui
    double erro_minimo = 1.0;//erro minimo começa com 1 (o maximo)

    for (int f = 0; f < NUM_FEATURES; f++) {//itera sobre cada caracteristicas (features)
        for (int i = 0; i < data->num_amostras; i++) {//itera sobre cada amostra do dataset
            double limiar_atual = data->features[i][f];//aqui o limiar é o valor da feature f da amostra i que vai ser testado (um "limite")
            for (int direcao = -1; direcao <= 1; direcao += 2) {//teste dois possiveis casos, direcao = -1 ou direcao = 1
                double erro_atual = 0.0;//define o erro para 0 pois vai começar a somar ele
                for (int j = 0; j < data->num_amostras; j++) {//itera sobre todas as amostras j para calcular o erro - através dele detecta se é um bom stum
                    double predicao; //define a predicao do stump (-1 ou 1) para a amostra j
                    // SUBSTITUIÇÃO DO TERNÁRIO ANINHADO
                    if (direcao == 1) {
                        //se for o primeiro caso, se a feature f da amostra j for >= que o limiar atual: predicao=1, se nao =-1
                        if (data->features[j][f] >= limiar_atual) {
                            predicao = 1.0;
                        } else {
                            predicao = -1.0;
                        }
                    } else {
                        //mesma logica para direção =1
                        if (data->features[j][f] < limiar_atual) {
                            predicao = 1.0;
                        } else {
                            predicao = -1.0;
                        }
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
    inicia_pesos(sample_weights, train_data->num_amostras);//inicia os pesos iniciais distribuidos pela quantidade de amostras
    *num_stumps = 0;//quantidade de stumps inicial é igual a 0 ->nenhuma arvore de decisao foi criada ainda

    for (int t = 0; t < ITERACOES; t++) {//vai iterando pela quantidade maxima de iterações (stumps)
        WeakLearner current_stump = define_melhor_stump(train_data, sample_weights);//acha a melhor arvore de decisao utilizando os pesos respectivos
        double error = 0.0;//inicializa o erro ponderado =0 (vai ser calculado depois para medir a eficacia do stump)

        for (int i = 0; i < train_data->num_amostras; i++) {
            double h_x; //calcula a predicao (h_x) de acordo com o limiar do melhor stump
            // SUBSTITUIÇÃO DO TERNÁRIO ANINHADO (1/2)
            if (current_stump.direcao == 1) {
                if (train_data->features[i][current_stump.indice_caracteristica] >= current_stump.limiar) {
                    h_x = 1.0;
                } else {
                    h_x = -1.0;
                }
            } else {
                if (train_data->features[i][current_stump.indice_caracteristica] < current_stump.limiar) {
                    h_x = 1.0;
                } else {
                    h_x = -1.0;
                }
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

        double sum_weights = 0.0;//define a soma dos pesos para começar a normalização (qual peso vai aumentar e qual vai diminuir)

        for (int i = 0; i < train_data->num_amostras; i++) {//itera sobre todas as amostras
            double h_x; //recalcula a mesma predicao feita antes
            // SUBSTITUIÇÃO DO TERNÁRIO ANINHADO (2/2)
            if (current_stump.direcao == 1) {
                if (train_data->features[i][current_stump.indice_caracteristica] >= current_stump.limiar) {
                    h_x = 1.0;
                } else {
                    h_x = -1.0;
                }
            } else {
                if (train_data->features[i][current_stump.indice_caracteristica] < current_stump.limiar) {
                    h_x = 1.0;
                } else {
                    h_x = -1.0;
                }
            }

            sample_weights[i] *= exp(-current_stump.alpha * train_data->rotulos[i] * h_x);/*ponto principal: amostras mal classificadas tem peso aumentado
            (o expoente vai ser maior que 0 - vai ser mais importante), se foi bem classificada tem peso diminuido (expoente menor que 0 -  vai ser dado menos importancia)*/
            sum_weights += sample_weights[i];//soma o novo peso para futura normalização (a soma dos pesos deve ser igual a 0 -> lembrar funcao inicia_pesos)
        }

        for (int i = 0; i < train_data->num_amostras; i++) {
            sample_weights[i] /= sum_weights;//normalização dos pesos (soma=1)
        }


        ensemble[*num_stumps] = current_stump;//adiciona o stump pra um pedaço do megazord
        (*num_stumps)++;//aumenta o numero de stumps que o ensemble vai ter
        printf("Iteracao %d: Alpha=%.4f, Erro=%.4f\n", t + 1, current_stump.alpha, error);//imprime o resultado para ver como estao indo os stumps
    }
    free(sample_weights);//libera a memoria alocada inicialmente
    return ensemble;//retorna o megazord com os classificadores fracos
}

// =========================================================================
// 4. AVALIAÇÃO E MÉTRICAS
// =========================================================================

double erro_previsao(int TP, int TN, int FN, int FP) {
    int total = TP + TN + FP + FN;
    // SUBSTITUIÇÃO DO TERNÁRIO: (total == 0) ? 0.0 : (double)(FP + FN) / total
    if (total == 0) {
        return 0.0;
    } else {
        return (double)(FP + FN) / total;
    }//formual do erro
}
double precisao(int TP, int FP) {
    // SUBSTITUIÇÃO DO TERNÁRIO: (TP + FP == 0) ? 0.0 : (double)TP / (TP + FP)
    if (TP + FP == 0) {
        return 0.0;
    } else {
        return (double)TP / (TP + FP);
    }//formula da precisao
}
double recall(int TP, int FN) {
    // SUBSTITUIÇÃO DO TERNÁRIO: (TP + FN == 0) ? 0.0 : (double)TP / (TP + FN)
    if (TP + FN == 0) {
        return 0.0;
    } else {
        return (double)TP / (TP + FN);
    }//formula do recall
}
double f1_score(double precision, double recall) {
    // SUBSTITUIÇÃO DO TERNÁRIO: (precision + recall == 0) ? 0.0 : (2.0 * precision * recall) / (precision + recall)
    if (precision + recall == 0) {
        return 0.0;
    } else {
        return (2.0 * precision * recall) / (precision + recall);
    }//formula do f1score
}

double classifica_amostra(double* sample_features, WeakLearner* ensemble, int num_stumps) {//aqui ele classifica a amostra (feature)
    double final_score = 0.0;
    for (int t = 0; t < num_stumps; t++) {//itera sobre todos os stumps criados
        double h_x;
        // SUBSTITUIÇÃO DO TERNÁRIO ANINHADO (h_x)
        if (ensemble[t].direcao == 1) {
            if (sample_features[ensemble[t].indice_caracteristica] >= ensemble[t].limiar) {
                h_x = 1.0;
            } else {
                h_x = -1.0;
            }
        } else {
            if (sample_features[ensemble[t].indice_caracteristica] < ensemble[t].limiar) {
                h_x = 1.0;
            } else {
                h_x = -1.0;
            }
        }
        final_score += ensemble[t].alpha * h_x;//vai somando a quantidade de votos de cada stump SOBRE A FEATURE EM DESTAQUE
    }
    
    // SUBSTITUIÇÃO DO TERNÁRIO FINAL (return)
    if (final_score >= 0.0) {
        return 1.0;
    } else {
        return -1.0;
    }//se o stump for positivo -> nao tem parkinson, se é negativo, tem parkinson
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
            { found = 1; break; }//checa se o id ja esta na lista
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
    printf("Matriz de Confusao (Pacientes - Total: %d):\n", unique_count);
    printf("Real +1 (Parkinson) | TP: %d | FN: %d\n", TP, FN);
    printf("Real -1 (Saudavel)  | FP: %d | TN: %d\n", FP, TN);

    double acc = (double)correct / unique_count;
    double prec = precisao(TP, FP);
    double rec = recall(TP, FN);
    double f1 = f1_score(prec, rec);

    printf("Acuracia: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f\n", acc, prec, rec, f1);//mostra todos as metricas correspondentes
}

// =========================================================================
// MAIN
// =========================================================================

int main() {
    setlocale(LC_ALL, "Portuguese");
    printf("=== PROJETO MACHINE LEARNING: ADABOOST.=== \n Eduardo Scalcon e Pedro Pilato\n");

    
    preparar_arquivos_treino_teste();//gera os dois arquivos separados para treino e validação

    
    printf("\n--- 2. Carregando Treino (%s) ---\n", ARQ_TREINO);
    Dataset* train_data = carrega_arquivo(ARQ_TREINO);//carrega e lê o arquivo de treino
    if (!train_data) { printf("Falha ao carregar treino.\n"); return 1; }

    
    printf("--- 3. Carregando Teste (%s) ---\n", ARQ_VALIDACAO);
    Dataset* test_data = carrega_arquivo(ARQ_VALIDACAO);//carrega e lê o arquivo de validação
    if (!test_data) { printf("Falha ao carregar teste.\n"); return 1; }

    
    int num_stumps = 0;
    printf("\n--- 4. Iniciando Treinamento AdaBoost ---\n");//começa de fato o treinamento do adaboost, utilizando arvores de decisão
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
