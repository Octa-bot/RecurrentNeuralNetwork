% 1. Preprocesamiento de datos
% Cargar datos de ejemplo (ajusta la ruta según la ubicación de tus datos)

folder = 'Señales P3';  % replace with the path to your folder
files = dir(fullfile(folder, '*.mat'));  % get a list of all .mat files in the folder
% files = natsortfiles(files); %alphanumeric sort
% [~, reindex] = sort( str2double( regexp( {files.name}, '\d+', 'match', 'once' )));
% files = files(reindex) ;
files = files(randperm(length(files)));
num_files = numel(files);  % get the number of files

for i = 1:num_files
    filename = fullfile(folder, files(i).name);  % get the full path to the file
    data(i) = load(filename);  % load the data from the file
    % do something with the data here, e.g.:
    disp(['Loaded file ' files(i).name])
end

% Obtener una derivación del ECG y normalizarla
fs = 500;
fs_new = 250;
[P, Q] = rat(fs_new / fs);
ecg = []; ecg_2 = [];
validation = []; validation2 = [];

for i=1:num_files
    ecg_lead = data(i).val(1, :);
    ecg_lead = ecg_lead - mean(ecg_lead);
    ecg_lead = ecg_lead / std(ecg_lead);

    ecg_lead2 = data(i).val(2, :);
    ecg_lead2 = ecg_lead2 - mean(ecg_lead2);
    ecg_lead2 = ecg_lead2 / std(ecg_lead2);

    % Remuestrear la señal a 250 Hz
    ecg_resampled = resample(ecg_lead, P, Q);
    ecg_resampled2 = resample(ecg_lead2, P, Q);
    if(i<=10)
        ecg = [ecg ecg_resampled];
        ecg_2 = [ecg_2 ecg_resampled2];
    else
        validation = [validation ecg_resampled];
        validation2 = [validation2 ecg_resampled2];
    end
end


% Aquí puedes agregar más señales a la matriz ecg_resampled, donde cada columna es una señal diferente
%%ecg_resampled = [ecg_resampled1, ecg_resampled2, ecg_resampled3, ...];

% Dividir los datos en conjuntos de entrenamiento y prueba
train_ratio = 0.8;
train_samples = floor(size(ecg, 2) * train_ratio);
X_train = ecg(1, 1:train_samples);
X_test = ecg(1,train_samples+1:end);

test_samples = floor(size(ecg_2, 2) * train_ratio);
Y_train = ecg_2(1,1:test_samples);
Y_test = ecg_2(1,test_samples+1:end);

% Convertir los datos a celdas (necesario para las redes LSTM en MATLAB)
X_train = num2cell(X_train, 1);
X_test = num2cell(X_test, 1);
Y_train = num2cell(Y_train, 1);
Y_test = num2cell(Y_test, 1);

% 2. Crear y entrenar la red LSTM
% Definir la arquitectura de la LSTM 
% 'numFeatures' es el número de características de entrada. 
% En este caso, se establece en 1 porque solo se procesa una señal de ECG a la vez.
%'numResponses' es el número de respuestas de la red. 
%En este caso, se establece en 1 porque la red LSTM produce
%una señal filtrada que tiene un solo valor en cada punto de tiempo.
%'numHiddenUnits' es el número de unidades ocultas en la capa LSTM. Este 
% valor define la complejidad de la red y puede ajustarse para mejorar el 
% rendimiento.
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 100;
%layers define la arquitectura de la red LSTM. En este caso, consiste en 
% una capa de entrada de secuencia, una capa LSTM, una capa completamente 
% conectada y una capa de regresión.
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Opciones de entrenamiento 
% options especifica las opciones de entrenamiento para la red LSTM. 
% En este caso, se utilizan las siguientes opciones:

%     Optimizador Adam para el entrenamiento.
%     100 épocas máximas.
%     Un tamaño de mini-lote de 128.
%     Una tasa de aprendizaje inicial de 0.01.
%     Un programa de tasa de aprendizaje 'piecewise', que reduce la tasa de aprendizaje en función de los factores y períodos especificados.
%     Un factor de caída de la tasa de aprendizaje de 0.5.
%     Un período de caída de la tasa de aprendizaje de 50 épocas.
%     Un umbral de gradiente de 1 para evitar que los gradientes exploten durante el entrenamiento.
%     Mezclar los datos en cada época.
%     Sin salida detallada del proceso de entrenamiento (Verbose = 0).
%     Gráficos de progreso del entrenamiento en tiempo real.
% nadam = nadamOptimize('LearnRate', 0.001, 'GradientDecayFactor', 0.9, 'SquaredGradientDecayFactor', 0.999);
options = trainingOptions( 'adam' ,...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 500, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'GradientThreshold', 0.2, ...
    'Shuffle', 'never', ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% Entrenar la LSTM
net = trainNetwork(X_train, Y_train, layers, options);

% 3. Evaluar y probar la red LSTM
% Probar la LSTM en los datos de prueba
YPred = predict(net, X_test);
ypred2=cell2mat(YPred);
x_test2=cell2mat(X_test);
YTEST=cell2mat(Y_test);
YPred = YPred';
% Calcular el error cuadrático medio en el conjunto de prueba
mse = mean(cellfun(@(x, y) mean((x - y).^2), YPred, Y_test));

% Mostrar el error cuadrático medio
fprintf("El error cuadrático medio en el conjunto de prueba es: %f", mse);

% Visualizar las señales de ECG originales, ruidosas y filtradas
num_signals = 3;
signal = size(ecg_resampled,2);
for i = 1:num_signals
    YPred = predict(net, validation((signal*(i-1)+1):(signal*i)));
    YPRED = num2cell(YPred,1);
    YPred = double(YPred);
    VAL2 = num2cell(validation2, 1);
    figure;
    mse = mean(cellfun(@(x, y) mean((x - y).^2), YPRED,VAL2((signal*(i-1)+1):(signal*i))));
    sgtitle(sprintf('Error de validacion #%i = %f', i, mse));
    subplot(3, 1, 1);
    plot(validation2((signal*(i-1)+1):(signal*i)));
    title(sprintf('Señal filtrada original %d', i));
    subplot(3, 1, 2);
    plot(validation((signal*(i-1)+1):(signal*i)));
    title(sprintf('Señal ruidosa %d', i));
    subplot(3, 1, 3);
    plot(YPred);
    title(sprintf('Señal filtrada red %d', i));
end

