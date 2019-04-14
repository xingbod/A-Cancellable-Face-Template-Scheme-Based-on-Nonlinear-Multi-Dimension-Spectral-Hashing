function template =  bloomfilter(feat)
N_BLOCKS = 1;% # number of blocks the facial image is divided into, also for LGBPHS algorihtm

N_HIST = 64 ; %# parameters fixed by LGBPHS
N_BINS = 7;

THRESHOLD = 0; % # binarization threshold for LGBPHS features

N_BITS_BF = 8 ; %# parameters for BF extraction
N_WORDS_BF = 4;
N_BF_Y = N_HIST/N_BITS_BF;
N_BF_X = (N_BINS+1)/N_WORDS_BF;
BF_SIZE = pow2(2, N_BITS_BF);

%     '''Extracts BF protected template from an unprotected template'''
template = zeros(N_BLOCKS , N_BF_X , N_BF_Y);

index = 1;
for i =1:N_BLOCKS
    block = feat(i,:);
    block = reshape(block, [N_HIST, N_BINS + 1]) ;% # add column of 0s -> now done on features!
    
    block = (block > THRESHOLD);
    
    for x =1:N_BF_X
        for y =1:N_BF_Y
            bf = zeros(1,BF_SIZE);
            
            ini_x = x * N_WORDS_BF;
            fin_x = (x + 1) * N_WORDS_BF;
            ini_y = y * N_BITS_BF;
            fin_y = (y + 1) * N_BITS_BF;
            new_hists = block(ini_y: fin_y, ini_x: fin_x);
            
            for k =1:N_WORDS_BF
                hist = new_hists(:, k);
                str="";
                for indx=1:length(hist)
                    str=str+num2str(hist(indx));
                end
                %                     location = int('0b' + ''.join([str(a) for a in hist]), 2)
                location = bin2dec(str);
                bf(location) = 1;
            end
            template(index) = bf;
            index =index+ 1;
        end
    end
end
