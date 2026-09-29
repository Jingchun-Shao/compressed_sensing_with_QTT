%% -----------------------------------------------------------
%%  Parameters
%% -----------------------------------------------------------
d            = 10;
n            = 2;
tt_ranks     = 1:10;  % Tensor ranks from 1 to 5

N        = n^d;
N_qtt    = n * ones(1,d);
num_rep  = 1000;
M_list   = 10:10:250;

F = dftmtx(N);          % unitary Fourier matrix

% Arrays: mean error + sample variance for each rank
meanErrTT      = zeros(numel(tt_ranks), numel(M_list));
varTT          = zeros(numel(tt_ranks), numel(M_list));

%% ----------------------------------------------------------------------
%%  A)  Ground truth = random rank-r TT tensor for each rank in 1:5
%% ----------------------------------------------------------------------
for r_idx = 1:numel(tt_ranks)
    tt_rank = tt_ranks(r_idx);
    disp("rank is: " + tt_rank);
    R        = [1, tt_rank*ones(1,d-1), 1];
    
    for m_idx = 1:numel(M_list)
        M = M_list(m_idx);
        e = zeros(1,num_rep);

        for rep = 1:num_rep
            g_qtt = tt_random(N_qtt,d,R);  % Generate rank-tt_rank tensor using tt_random
            g     = full(g_qtt);  % Convert tensor to full format

            idx = randi(N, M, 1);         % random Fourier rows
            G   = F(idx,:)/sqrt(M);
            y   = G*g;
            e(rep) = abs( norm(y)^2 / norm(g)^2 - 1 );   % δ(g) error
        end

        meanErrTT(r_idx, m_idx) = mean(e);
        varTT(r_idx, m_idx)     = var(e,1);       % population variance
    end
end

%% ------------------------------  PLOT 1: mean error -------------------
figure; hold on; box on;
for r_idx = 1:numel(tt_ranks)
    plot(M_list, meanErrTT(r_idx,:), 'o-','LineWidth',1.2,'MarkerSize',6,...
         'DisplayName',sprintf('TT (rank %d)',tt_ranks(r_idx)));
end

Cref = meanErrTT(1,1);
plot(M_list, Cref*(M_list(1)./M_list).^0.5,'k--','LineWidth',1.5,...
     'DisplayName','M^{-1/2}');
% plot(M_list, Cref*(M_list(1)./M_list), 'k:','LineWidth',1.5,...
%      'DisplayName','M^{-1}');

set(gca,'XScale','log','YScale','log');
xlabel('measurements  M');
ylabel('mean error  E[|δ(g)|]');
title(sprintf('d = %d   |   %d trials   |   mean error', d, num_rep));
legend('Location','southwest'); grid on;

% %% ------------------------------  PLOT 2: variance ---------------------
% figure; hold on; box on;
% for r_idx = 1:numel(tt_ranks)
%     plot(M_list, varTT(r_idx,:), 'o-','LineWidth',1.2,'MarkerSize',6,...
%          'DisplayName',sprintf('Var–TT (rank %d)',tt_ranks(r_idx)));
% end
% 
% % -------- reference line  y = C / M  (slope –1 in log–log) -------------
% C_ref  = varTT(1,1) * M_list(1);
% refLine = C_ref ./ M_list;
% plot(M_list, refLine, 'k--', 'LineWidth', 1.4, ...
%      'DisplayName', sprintf('C/M  (C = %.2g)', C_ref));
% 
% set(gca,'XScale','log','YScale','log');
% xlabel('measurements  M');  ylabel('error variance');
% title(sprintf('d = %d   |   %d trials   |   variance of error', d, num_rep));
% legend('Location','southwest'); grid on;
