%% Load data
 clear;
 clc;
filepath = '.\source\';
dirOutput = dir(fullfile(filepath,'source*'));
filename = {dirOutput.name};
Length_Names = size(filename,2);

for k = 1 : Length_Names  % sometimes need to manually change k, e.g. k = 91:Length_Names
    path = strcat(filepath, filename(k));
    data = load(path{1});

    temp = regexp(filename(k), '_', 'split');
    A_p = temp{1}{2};
    sizeInterV = str2num(A_p);
    nodes = size(sizeInterV,2);
    nSamples = size(data,1);
    

    % Multiset CCA             
    R = cov(data); % data: sample * nodes
    for ii=nodes:-1:1
            j=(1:sizeInterV(ii))+sum(sizeInterV(1:ii-1)); 
            D(j,j)=R(j,j); 
    end
    [V,~]=eig(R,D);
    for jj = 1:length(V)
            if sum(V(:,jj)==0)==0
                    A = V(:,jj); 
            end
    end
    if isempty(A)
            A = V(:,1)+eps;
    end

    % canonical variables Y
    Y = zeros(nSamples,nodes);
    kk = 1;
    for ii = 1:length(sizeInterV)                
            Y(:,ii) = data(:,kk:kk+sizeInterV(ii)-1)*A(kk:kk+sizeInterV(ii)-1);
            kk = kk+sizeInterV(ii);
    end
            
    name = strrep(filename(k),'source','cca');
    savepath = ['.\cca\',name{1}];
    save(savepath,'Y','-ascii');
end