%% Note
% Before running the algorithms you have to compile the .cpp files, using
% mexAll.m

clear;
mexAll;

reg_mode = 2;
gen_model = 0;
%% Get the data X, y
%   1. X: [m*n], each column of X is one sample data;
%   2. y: [n*1], is the label of each sample data.A(i,:).
%   3. w: [m*1], is the number of features.

if gen_model
    model_gen_para.size_of_features = 10;
    model_gen_para.size_of_data = 100;
    n = 100;
    d = 10;
    model_gen_para.noise = 0.1;
    [ X, y, true_w ] = Model_Gen_linear( model_gen_para );
else
    load('../data/rcv1_train.binary.mat');
    load('../data/adult.mat');
    X = [ones(size(X,1),1) X];
    [n, d] = size(X);
    X = X';
    
%     addpath('..\libsvm');
%     [y, X] = libsvmread('..\data\australian');

    addpath('../libsvm-3.21/matlab');
%     [y, X] = libsvmread('../data/australian');
    [y, X] = libsvmread('../data/ijcnn1.t');
    [n, d] = size(X);
    X = X';    
    X = full(X); 

    % Data normalization 
    sum1 = 1./sqrt(sum(X.^2, 1));
    if abs(sum1(1) - 1) > 10^(-10)
        X = X.*repmat(sum1, d, 1);
    end
end

%% Get the approximation of the best parameter
% lambda = 1/(n);
lambda = 1 / 10000;
Lmax   = (0.25 * max(sum(X.^2,2)) + lambda);


%% Test l2
if (reg_mode == 2)
    % if set to false for not computing function values in process
    % if set to true, all algorithms compute function values and display them
    history = true;
    
    % SVRG
    outer_loops = 50;
    inner_loops = 2*n;

    w_SVRG = zeros(d, 1);
    tic;
    if (history)
        histSVRG_l2 = Alg_SVRG(w_SVRG, X, y, lambda, Lmax, outer_loops);
    else
        Alg_SVRG(w_SVRG, X, y, lambda, Lmax, outer_loops);
    end
    time_SVRG = toc;
    fprintf('Time spent on SVRG: %f seconds \n', time_SVRG);
    xSVRG_l2 = 0:outer_loops;

    
    % SAGA 
    outer_loops = 100;
    iVals = int64(floor(n*rand(outer_loops*n,1)));

    w_SAGA = zeros(d, 1);
    tic;
    if (history)
        histSAGA_l2 = Alg_SAGA(w_SAGA, X, y, lambda, Lmax, iVals, outer_loops);
    else
        Alg_SGAG(w_SAGA, X, y, lambda, Lmax, iVals, outer_loops);
    end
    time_SAGA = toc;
    fprintf('Time spent on SAGA: %f seconds \n', time_SAGA);
    xSAGA_l2 = 0:outer_loops/2;
    
    
    % Katyusha
    outer_loops = 50;
    w_Katyusha = zeros(d, 1);
    tau  = min(0.5, sqrt(1.25*n*lambda/(3*Lmax)));
    
    tic;
    if (history)
        histKatyusha_l2 = Alg_Katyusha(w_Katyusha, full(X), y, lambda, 3*Lmax, tau, outer_loops);
    else
        Alg_Katyusha(w_Katyusha, full(X), y, lambda, 3*Lmax, tau, outer_loops);
    end
    time_Katyusha = toc;
    fprintf('Time spent on Katyusha: %f seconds \n', time_Katyusha);
    xKatyusha_l2 = 0:outer_loops;
    
end

%% Test l1
if (reg_mode == 1)
    % if set to false for not computing function values in process
    % if set to true, all algorithms compute function values and display them
    history = true;

    outer_loops = 100;
    inner_loops = 2*n;

    w_SVRG = zeros(d, 1);
    tic;
    if (history)
        histSVRG_l1 = Alg_prox_SVRG(w_SVRG, X, y, lambda, Lmax, outer_loops);
    else
        Alg_prox_SVRG(w_SVRG, X, y, lambda, Lmax, outer_loops);
    end
    time_SVRG = toc;
    fprintf('Time spent on SVRG: %f seconds \n', time_SVRG);
    xSVRG_l1 = 0:outer_loops;
    
    outer_loops = 100;
    iVals = int64(floor(n*rand(outer_loops*n,1)));

    w_SAGA = zeros(d, 1);
    tic;
    if (history)
        histSAGA_l1 = Alg_prox_SAGA(w_SAGA, X, y, lambda, Lmax, iVals, outer_loops);
    else
        Alg_prox_SAGA(w_SAGA, X, y, lambda, Lmax, iVals, outer_loops);
    end
    time_SAGA = toc;
    fprintf('Time spent on SAGA: %f seconds \n', time_SAGA);
    xSAGA_l1 = 0:outer_loops/2;
    

    outer_loops = 50;
    w_Katyusha = zeros(d, 1);
    tau  = min(0.5, sqrt(1.25*n*lambda/(3*Lmax)));
    
    tic;
    if (history)
        histKatyusha_l1 = Alg_Katyusha(w_Katyusha, full(X), y, lambda, 3*Lmax, tau, outer_loops);
    else
        Alg_Katyusha(w_Katyusha, full(X), y, lambda, 3*Lmax, tau, outer_loops);
    end
    time_Katyusha = toc;
    fprintf('Time spent on Katyusha: %f seconds \n', time_Katyusha);
    xKatyusha_l1 = 0:outer_loops; 
end



deleteMex

%% Plot the results

if(history)
%     australian dataset
    fstar = 0.1369;
    % linear model generator; rng(23)
%     fstar = 0.039; 
    fEvals = cell(3, 1);
    fVals = cell(3, 1);
    if (reg_mode == 1)
        fEvals{1} = xSVRG_l1(1:1:xSVRG_l1(end));
        fEvals{2} = xSAGA_l1(1:1:xSAGA_l1(end));
        fEvals{3} = xKatyusha_l1(1:1:xKatyusha_l1(end));
        fVals{1} = histSVRG_l1(1:1:xSVRG_l1(end)) - fstar;
        fVals{2} = histSAGA_l1(1:1:xSAGA_l1(end)) - fstar;
        fVals{3} = histKatyusha_l1(1:1:xKatyusha_l1(end)) - fstar;

    end
    
    if (reg_mode == 2)
        fEvals{1} = xSVRG_l2(1:1:xSVRG_l2(end));
        fEvals{2} = xSAGA_l2(1:1:xSAGA_l2(end));
        fEvals{3} = xKatyusha_l2(1:1:xKatyusha_l2(end));
        fVals{1} = histSVRG_l2(1:1:xSVRG_l2(end)) - fstar;
        fVals{2} = histSAGA_l2(1:1:xSAGA_l2(end)) - fstar;
        fVals{3} = histKatyusha_l2(1:1:xKatyusha_l2(end)) - fstar;
    end    
    
    n = length(fVals);

    colors = colormap(lines(8)); colors = colors([1 2 3 4], :);
    lineStyle = cellstr(['-'; '-'; '-'; '-';'-'; '-';]);
    markers = cellstr(['s'; 'o'; 'p'; '*'; 'd'; 'x';]);
    markerSpacing = [3 3 3 5 3 3; 2 1 3 2 3 1]';
    names = cellstr(['SVRG    '; 'SAGA    '; 'KATYUSHA']);

    options.legendLoc = 'NorthEast';
    options.logScale = 2;
    options.colors = colors;
    options.lineStyles = lineStyle(1:n);
    options.markers = markers(1:n);
    options.markerSize = 12;
    options.markerSpacing = markerSpacing;
    options.legendStr = names;
    options.legend = names;
    options.ylabel = 'Objective minus Optimum';
    options.xlabel = 'Passes through Data';
    options.labelLines = 1;
    options.labelRotate = 1;
    options.xlimits = [];
    options.ylimits = [];

    prettyPlot(fEvals,fVals,options); % (Thanks, Mark Schmidt)

end

