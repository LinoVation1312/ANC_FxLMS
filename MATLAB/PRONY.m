%% IIR ANALYSIS - PRONY METHOD (Grid search on M,N)
clear; clc; close all;


pi_shift = 2*pi*20 ; %should be a 2*PI multiple, adjust the "0"
criteria = 0.001   ; %accuracy of the IIR model (criteria < 0.025 ; redo your measurement if not.)


%% === Load impulse response file ===
[filename, pathname] = uigetfile('*.mat', 'Select the .mat file containing measured impulse response data');

if isequal(filename, 0)
    error('❌ File loading cancelled by user.');
else
    fullpath = fullfile(pathname, filename);
    fprintf('📂 Loading file: %s\n', fullpath);
    data = load(fullpath);
end

% === Check required structure ===
if ~isfield(data, 'measurementData')
    error('❌ The file must contain a variable named "measurementData".');
end

measurementData = data.measurementData;

%% === Extract IR ===

h_full = measurementData.ImpulseResponse(2,1).Amplitude;

t_sec  = measurementData.ImpulseResponse(1,1).Time;
fs     = measurementData.SampleRate(1,1);

loop_back=measurementData.ImpulseResponse(1,1).Amplitude;



[~, latency] = max(abs(loop_back));
latency_s=latency/fs;
latency_ms=latency_s*1000;

figure;
subplot(211);plot(loop_back,'LineWidth',3);xlim([latency-250 latency+250]);title('LOOPBACK MEASUREMENT')
subplot(212);plot(t_sec*1000, h_full,'LineWidth', 3);xlabel('Time (ms)');ylabel('Amplitude');title('Full Impulse Response');
grid on;
xlim([0 max(t_sec)*1000])
%% Remove initial system latency (in samples)
h_full = h_full(latency + 1:end);
t_sec  = t_sec(1:end-latency);

%% frequency-domain data (magnitude and phase)
f = measurementData.MagnitudeResponse.Frequency;       % Hz
Mag_dB_measured = measurementData.('MagnitudeResponse')(2,1).MagnitudeDB;
phase_measured  = measurementData.('PhaseResponse')(2,1).Phase+(2*pi*latency_s.*f)+pi_shift;

%% Trimming on the first peak of the IR

[h_cut, tau, tau_samples, idx_trim_start] = automatic_ir_trimming(h_full, fs);

tau_ms=tau*1000;

fprintf("Trim starts at sample %d (%.2f ms)\n", idx_trim_start, tau_ms);

%% Energy-based IR trimming (trimming after 99 or 99.9% of the energy)

% Duration constraints (in milliseconds)
min_duration_ms = 80;
max_duration_ms = 300;

[h_trimmed, idx_trim, energy_threshold] = energy_based_trimming(h_cut, fs, min_duration_ms, max_duration_ms);

fprintf('✅ Trimmed IR with %.3f energy threshold and duration in [%d...%d] ms → %d samples\n', ...
    energy_threshold, min_duration_ms, max_duration_ms, idx_trim);

%% === Grid search over IIR orders (Prony method) ===

max_order = 15;
results = [];

for N = 1:max_order
    for M = 1:N
        try
            [b, a] = prony(h_trimmed, M, N);
            poles = roots(a);
            is_stable = all(abs(poles) < 1);

            if is_stable
                % Simulate the IR from the model
                h_model = filter(b, a, [1; zeros(length(h_trimmed) - 1, 1)]);

                % Compute time-domain NMSE
                nmse_ir = sum((h_trimmed - h_model).^2) / sum(h_trimmed.^2);

                % Frequency-domain model response
                Nfft = 2^nextpow2(length(h_trimmed) * 4);
                H_model_full = fft(h_model, Nfft);

                freq_fft = (0:Nfft - 1) * (fs / Nfft);

                % Convert measured magnitude to linear scale
                H_orig = 10.^(Mag_dB_measured / 20);

                % Interpolate model magnitude on the measured freq grid
                mag_model = abs(H_model_full);
                mag_model_interp = interp1(freq_fft, mag_model, f, 'linear', 'extrap');

                % Focus on 50–250 Hz band
                freq_mask = (f >= 50) & (f <= 250);
                H_orig_filtered = H_orig(freq_mask);
                mag_model_filtered = mag_model_interp(freq_mask);

                % Frequency-domain NMSE
                nmse_freq = sum((H_orig_filtered - mag_model_filtered).^2) / ...
                            sum(H_orig_filtered.^2);

                % Time-domain correlation
                corr_coef = corrcoef(h_trimmed, h_model);
                correlation = corr_coef(1, 2);

                % Store result
                results = [results; struct(...
                    'M', M, 'N', N, ...
                    'nmse_ir', nmse_ir, ...
                    'nmse_freq', nmse_freq, ...
                    'corr', correlation, ...
                    'is_stable', true, ...
                    'b', b, 'a', a)];

            else
                % Unstable model
                results = [results; struct(...
                    'M', M, 'N', N, ...
                    'nmse_ir', NaN, ...
                    'nmse_freq', NaN, ...
                    'corr', NaN, ...
                    'is_stable', false, ...
                    'b', [], 'a', [])];
            end

        catch ME
            % Catch fitting errors
            warning('⚠️ Error at M=%d, N=%d: %s', M, N, ME.message);
            results = [results; struct(...
                'M', M, 'N', N, ...
                'nmse_ir', NaN, ...
                'nmse_freq', NaN, ...
                'corr', NaN, ...
                'is_stable', false, ...
                'b', [], 'a', [])];
        end
    end
end



%% Filter by NMSE < criteria and select model with lowest (M + N)

filtered = results([results.is_stable] & [results.nmse_ir] < criteria);

if isempty(filtered)
    fprintf('\n⚠️  No stable model with NMSE < %2f found.\n',criteria);
else
    % Compute total order (complexity)
    total_order = arrayfun(@(r) r.M + r.N, filtered);
    
    % Find minimum total order
    min_order = min(total_order);
    idx_candidates = find(total_order == min_order);
    
    % Among them, choose the one with smallest NMSE
    nmse_candidates = [filtered(idx_candidates).nmse_ir];
    [~, idx_best_local] = min(nmse_candidates);
    idx_best = idx_candidates(idx_best_local);
    best_model = filtered(idx_best);
    
    % Display filtered candidates
    fprintf('\n✅ Stable models with NMSE < 0.01:\n');
    fprintf('| %4s | %4s | %5s | %9s | %9s | %9s |\n', ...
        'M', 'N', 'M+N', 'NMSE_IR', 'NMSE_FREQ', 'Corr');
    fprintf('|%s|\n', repmat('-', 1, 67));
    
    for i = 1:length(filtered)
        fprintf('| %4d | %4d | %5d | %9.4g | %9.4g | %9.4f |\n', ...
            filtered(i).M, filtered(i).N, filtered(i).M + filtered(i).N, ...
            filtered(i).nmse_ir, filtered(i).nmse_freq, filtered(i).corr);
    end

    % Final best model
    fprintf('\n🌟 Best model (NMSE < %2f and min(M+N)):\n', criteria);
    fprintf('→ M = %d, N = %d (M+N = %d)\n', best_model.M, best_model.N, best_model.M + best_model.N);
    fprintf('→ NMSE_IR: %.4g | NMSE_FREQ: %.4g | Corr: %.4f\n', ...
        best_model.nmse_ir, best_model.nmse_freq, best_model.corr);
end


%% Best IR computation

t_trimmed = (0:length(h_trimmed)-1)/fs;

% Generate the impulse response of the IIR filter
impulse = [1; zeros(length(h_trimmed)-1, 1)];
h_model_1 = filter(best_model.b, best_model.a, impulse);

%% Delay !!
delay = [zeros(tau_samples, 1); 1];  % delayed Dirac impulse

% Convolve to apply the pure delay (without modifying IIR coefficients) ;)

h_model = conv(h_model_1, delay);
h_model = h_model(1:length(h_model_1));  % truncate to original lengt

%% Calculate modeled magnitude
Nfft = 2^nextpow2(length(h_trimmed)*4);
H_model = fft(h_model, Nfft);

freq_fft = (0:Nfft-1)*(fs/Nfft);
mag_model_dB = 20*log10(abs(H_model));

% Interpolate modeled magnitude on measurement freq grid
mag_model_interp = interp1(freq_fft, mag_model_dB, f, 'linear', 'extrap');

%% Calculate modeled phase
phase_model = unwrap(angle(H_model));                      % unwrap phase in radians
phase_model_interp = interp1(freq_fft, phase_model, f, 'linear', 'extrap');

%% Frequency band filtering between 35 Hz and 250 Hz
freq_mask = (f >= 50) & (f <= 250);

f_filtered = f(freq_mask);
Mag_dB_measured_filtered = Mag_dB_measured(freq_mask);
mag_model_interp_filtered = mag_model_interp(freq_mask);
phase_model_interp_filtered = phase_model_interp(freq_mask);%-(2*pi*f_filtered*tau);
phase_measured_filtered=phase_measured(freq_mask);

err=(phase_measured_filtered-phase_model_interp_filtered);

%% err calc.
% Masque fréquentiel entre 50 et 250 Hz
freq_mask_50_250 = (f_filtered >= 50) & (f_filtered <= 250);

% Extraire erreur sur cette bande
err_filtered = err(freq_mask_50_250);

% Moyenne et écart-type
mean_err = mean(err_filtered);
std_err = std(err_filtered);

% Affichage
fprintf('Mean error on the phase (50-250 Hz) : %.4f radians\n', mean_err);
fprintf('Standard deviation on the error phase (50-250 Hz) : %.4f radians\n', std_err);
figure;plot(f_filtered,err,'LineWidth',3)
title('Phase error : Measured - Modeled');
xlabel('Frequency (Hz)');
ylabel('Phase (rad)');
yline(pi/4,  'r--', 'LineWidth', 4); %  +π/4
yline(-pi/4, 'r--', 'LineWidth', 4); %  -π/4


%% Plot 

% Plot time-domain IR and frequency responses (magnitude + phase)

figure;

% 1 - Time-domain impulse response complète
subplot(2,2,1);
plot(t_sec*1000, h_full,'LineWidth', 3);
xlabel('Time (ms)');
ylabel('Amplitude');
title('Full Impulse Response');
grid on;
xlim([0 max(t_sec)*70])

t_shifted = t_trimmed * 1000 + tau_ms;

% 2 - Time-domain impulse response comparée (mesurée vs modèle)
subplot(2,2,3);
plot(t_shifted, h_trimmed, 'b-', 'LineWidth', 2.5); hold on;
plot(t_trimmed*1000, h_model, 'r--', 'LineWidth', 2.5);
xlabel('Time (ms)');
ylabel('Amplitude');
title(sprintf('Trimmed IR Comparison (M = %d, N = %d) — NMSE = %.4f', ...
    best_model.M, best_model.N, nmse_ir));
legend('Measured h_{trimmed}', 'Modeled h_{model}');
xlim([0 max(t_trimmed)*300]);  % durée totale IR en ms
grid on;

% 3 - Magnitude réponse en fréquence (log freq)
subplot(2,2,2);
semilogx(f_filtered, Mag_dB_measured_filtered, 'b-', 'LineWidth', 2.5); hold on;
semilogx(f_filtered, mag_model_interp_filtered, 'r--', 'LineWidth', 2.5);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title(sprintf('Magnitude Response (35–250 Hz) — NMSE = %.4f', nmse_freq));
legend('Measured', 'Modeled (Prony)');
xlim([50 250]);
grid on;

% 4 - Phase réponse en fréquence (rad)
subplot(2,2,4);
plot(f_filtered, phase_measured_filtered, 'b-', 'LineWidth', 2.5); hold on;
plot(f_filtered, phase_model_interp_filtered, 'r--', 'LineWidth', 2.5);
xlabel('Frequency (Hz)');
ylabel('Phase (rad)');
title('Phase Response Comparison (35 Hz - 250 Hz)');
legend('Measured', 'Modeled (Prony)');
xlim([50 250]);
grid on;

%%
% --- Retrieve excitation and measured signals ---
x = measurementData.RawAudioData(2,1).ExcitationSignal(:);
y_measured = measurementData.RawAudioData(2,1).RecordedSignal(:);

% --- Apply the IIR model filter to the excitation signal ---
y_model = filter(best_model.b, best_model.a, x);

% --- Check for valid (positive) latency and tau before applying alignment ---
if latency > 0 && tau > 0
    % Align measured signal by removing latency and tau samples
    start_idx = latency + tau_samples;
    y_measured_aligned = y_measured(start_idx:end);

    % Zero-padding at the end to preserve the original signal length
    y_measured_aligned(end+1:end+(start_idx-1)) = 0;
else
    error('Non-physical latency or tau detected (latency = %d, tau = %d)', latency, tau_samples);
end

% --- Truncate signals to the same length ---
len = min(length(y_measured_aligned), length(y_model));
y_measured_aligned = y_measured_aligned(1:len);
y_model = y_model(1:len);


% --- Plot downsampled signals for clarity ---
downsample_factor = 1;
idx = 1:downsample_factor:len;

% --- Compute and display NMSE ---
nmse_conv = sum((y_measured_aligned - y_model).^2) / sum(y_measured_aligned.^2);
fprintf('NMSE between modeled and measured output: %.4f\n', nmse_conv);


figure;
plot(idx/fs, y_measured_aligned(idx), 'k', 'DisplayName', 'Measured signal (aligned)'); hold on;
plot(idx/fs, y_model(idx), 'r--', 'DisplayName', 'Model (filtered output)');
xlabel('time (s)');
ylabel('Amplitude');
legend('Location','best');
grid on;
title(sprintf('Measured vs. IIR Model Output — NMSE = %.4f, M=%d, N=%d', nmse_conv,best_model.M, best_model.N));
xlim([0 max(idx/fs)]);



[corr_xy, lags] = xcorr(y_measured_aligned, y_model);
[~, idx_max] = max(abs(corr_xy));
delay_samples = lags(idx_max);

% --- Récupération des coeffs ---
b = best_model.b;
a = best_model.a;

fprintf('\n📐 Best Model Coeffs :\n');
fprintf('b = ['); fprintf('%.6f, ', b(1:end-1)); fprintf('%.6f]\n', b(end));
fprintf('a = ['); fprintf('%.6f, ', a(1:end-1)); fprintf('%.6f]\n', a(end));
fprintf('tau (ms) = %.2f',tau_ms);

 
%% === DIALOGUE : Save coefficients in .json format ===
resp = questdlg('Do you want to save coefficients in .json format?', ...
    'Save .json', 'Yes', 'No', 'Yes');

if strcmp(resp, 'Yes')
    [filename, pathname] = uiputfile('filter_coeffs_RAW.json', 'Choose .json filename');
    if ischar(filename)
        coeffs_struct.B = b;             
        coeffs_struct.A = a;               
        coeffs_struct.tau_ms = tau_ms; 
        
        jsonStr = jsonencode(coeffs_struct);
        fid = fopen(fullfile(pathname, filename), 'w');
        fwrite(fid, jsonStr, 'char');
        fclose(fid);
        
        fprintf('✅ JSON file saved: %s\n', fullfile(pathname, filename));
    end
end
