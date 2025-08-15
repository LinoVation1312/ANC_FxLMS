clc, close all, clear all;

%% System Parameters
fs = 8820;                 % Sampling frequency
f_low = 35;                % Low frequency of the filter
f_high = 400;              % High frequency of the filter 
L = 128;                   % Adaptive filter length
mu = 3e-2;                 % Adaptation step size
chunk_size =32;          % Chunk size for real-time simulation

%% Secondary Path (loaded from JSON)
json_data = fileread('TESTANC/filter_coeffs.json');
filter_coeffs = jsondecode(json_data);
sec_path.B = filter_coeffs.B;
sec_path.A = filter_coeffs.A;
tau_ms = filter_coeffs.tau_ms;

% Secondary path stability check
if any(abs(roots(sec_path.A)) >= 1)
    error('The secondary path model is unstable!');
end

%% Delay calculation
d = round(tau_ms * fs / 1000); % Delay in samples
fprintf('Secondary path delay: %d samples\n', d);

%% Signal Generation

wav_filepath = 'bruit_avant_anc.wav'; 
[noise_prim_raw, fs_wav] = audioread(wav_filepath);

% Sampling frequency verification
if fs_wav ~= fs
    fprintf('Resampling file from %d Hz to %d Hz...\n', fs_wav, fs);
    noise_prim_raw = resample(noise_prim_raw, fs, fs_wav);
end

% Ensure signal is mono (take first channel if stereo)
if size(noise_prim_raw, 2) > 1
    noise_prim_raw = noise_prim_raw(:, 1);
    disp('Audio file converted to mono.');
end

% Adjust simulation duration to audio file duration
nSamples = length(noise_prim_raw);
T = nSamples / fs;
t = (0:nSamples-1)'/fs;

% 1. Use audio file as primary noise
% Note: Amplitude is normalized. You can adjust if needed.
noise_prim = 3 * noise_prim_raw / max(abs(noise_prim_raw));

% 2. Broadband reference for ANC
REF = noise_prim;  % Use complete noise signal directly as reference

% 3. Primary path modeling (simplified, broadband)
% Noise arriving at error microphone is also broadband.
delay_prim_ms = 10; % Desired delay in ms
delay_prim_samples = round(delay_prim_ms * fs / 1000);
fprintf('Primary path delay: %d samples\n', delay_prim_samples);
noise_prim_delayed = [zeros(delay_prim_samples, 1); noise_prim(1:end-delay_prim_samples)];
d_prim = 0.7 * noise_prim_delayed; % Noise at error microphone

%% Buffer and state initialization
W = 0.01 * randn(L, 1);    % Random initialization (avoids zero weights)
buffer_x = zeros(L, 1);     % Buffer for reference signal
buffer_x_filt = zeros(L, 1);% Buffer for filtered reference (by complete secondary path)

% IIR filter states
state_filt_sec_phys = zeros(max(length(sec_path.A), length(sec_path.B))-1, 1);

%% Pre-computation of filtered reference (FXLMS) with delay included
x_filt_raw = filter(sec_path.B, sec_path.A, REF);
% Apply delay of d samples
x_filt = [zeros(d,1); x_filt_raw(1:end-d)]; 

%% Buffer initialization to delay physical filter output
buffer_y_filt = zeros(d,1);   % Buffer of size d to delay y_filt

%% FXLMS Simulation with chunk-based processing (real-time simulation)

y = zeros(nSamples, 1);     % Anti-noise signal
e = zeros(nSamples, 1);     % Error signal
e_mes = zeros(nSamples, 1); % Measured error

% Performance metrics initialization
chunk_times = [];           % Processing time of each chunk
chunk_count = 0;

fprintf('\n=== REAL-TIME CHUNK-BASED SIMULATION ===\n');
fprintf('Chunk size: %d samples (%.2f ms at %d Hz)\n', chunk_size, chunk_size*1000/fs, fs);

% Calculate number of complete chunks
num_chunks = floor(nSamples / chunk_size);
fprintf('Number of chunks to process: %d\n', num_chunks);
fprintf('Starting processing...\n\n');

% Chunk-based processing
for chunk_idx = 1:num_chunks
    chunk_start = (chunk_idx - 1) * chunk_size + 1;
    chunk_end = min(chunk_idx * chunk_size, nSamples);
    chunk_indices = chunk_start:chunk_end;
    
    % Timer to measure performance
    chunk_timer = tic;
    chunk_count = chunk_count + 1;
    
    % Process chunk sample by sample
    for n = chunk_indices
        % 1. Update reference buffer
        buffer_x = [REF(n); buffer_x(1:end-1)];
        
        % 2. Calculate anti-noise output
        y(n) = W' * buffer_x;
        
        % 3. Filter anti-noise signal through secondary path
        [y_filt, state_filt_sec_phys] = filter(sec_path.B, sec_path.A, y(n), state_filt_sec_phys);
        
        % 4. Delay management: store filtered output and take oldest
        if d > 0
            s_sec = buffer_y_filt(1);           % Oldest value (d samples ago)
            buffer_y_filt = [buffer_y_filt(2:end); y_filt]; % Shift and add new
        else
            s_sec = y_filt;
        end
        
        % 5. Error calculation (primary noise + delayed and filtered anti-noise effect)
        e(n) = d_prim(n) + s_sec;
        e_mes(n) = e(n);  % For analysis
        
        % 6. Update buffer for filtered reference (FXLMS)
        buffer_x_filt = [x_filt(n); buffer_x_filt(1:end-1)];
        
        % 7. Coefficient update (only after buffer filling)
        if n > d + L
            % NLMS Algorithm (Normalized Least Mean Squares)
            % Step is normalized by filtered reference signal power
            power = buffer_x_filt' * buffer_x_filt;
            mu_norm = mu / (power + 1e-6); % Use fixed mu and epsilon for stability
            
            % Update filter weights
            W = W - mu_norm * e(n) * buffer_x_filt;
        end
    end
    
    % Measure chunk processing time
    chunk_time = toc(chunk_timer);
    chunk_times = [chunk_times; chunk_time];
    
    % Display progress and real-time metrics
    if mod(chunk_idx, 50) == 0 || chunk_idx <= 10
        chunk_duration_ms = (chunk_end - chunk_start + 1) * 1000 / fs;
        processing_time_ms = chunk_time * 1000;
        real_time_factor = processing_time_ms / chunk_duration_ms;
        
        fprintf('Chunk %3d/%d | Duration: %5.2f ms | Processing: %5.2f ms | RTF: %.3f', ...
                chunk_idx, num_chunks, chunk_duration_ms, processing_time_ms, real_time_factor);
        
        if real_time_factor > 1.0
            fprintf(' ⚠ OVERRUN!\n');
        else
            fprintf(' ✓\n');
        end
    end
    
    % Real-time pause simulation (optional)
    % pause(chunk_duration_ms / 1000); % Uncomment for true real-time simulation
end

% Process remaining samples (if not multiple of chunk_size)
remaining_samples = nSamples - num_chunks * chunk_size;
if remaining_samples > 0
    fprintf('\nProcessing %d remaining samples...\n', remaining_samples);
    for n = (num_chunks * chunk_size + 1):nSamples
        % Same processing as in main loop
        buffer_x = [REF(n); buffer_x(1:end-1)];
        y(n) = W' * buffer_x;
        [y_filt, state_filt_sec_phys] = filter(sec_path.B, sec_path.A, y(n), state_filt_sec_phys);
        
        if d > 0
            s_sec = buffer_y_filt(1);
            buffer_y_filt = [buffer_y_filt(2:end); y_filt];
        else
            s_sec = y_filt;
        end
        
        e(n) = d_prim(n) + s_sec;
        e_mes(n) = e(n);
        buffer_x_filt = [x_filt(n); buffer_x_filt(1:end-1)];
        
        if n > d + L
            power = buffer_x_filt' * buffer_x_filt;
            mu_norm = mu / (power + 1e-6);
            W = W - mu_norm * e(n) * buffer_x_filt;
        end
    end
end

%% Real-time performance analysis
fprintf('\n=== PERFORMANCE ANALYSIS ===\n');
avg_chunk_time = mean(chunk_times) * 1000; % in ms
max_chunk_time = max(chunk_times) * 1000;  % in ms
chunk_duration_target = chunk_size * 1000 / fs; % theoretical chunk duration in ms

fprintf('Average processing time per chunk: %.2f ms\n', avg_chunk_time);
fprintf('Maximum processing time: %.2f ms\n', max_chunk_time);
fprintf('Theoretical chunk duration: %.2f ms\n', chunk_duration_target);
fprintf('Average real-time factor: %.3f\n', avg_chunk_time / chunk_duration_target);
fprintf('Maximum real-time factor: %.3f\n', max_chunk_time / chunk_duration_target);

overruns = sum(chunk_times * 1000 > chunk_duration_target);
fprintf('Number of overruns: %d/%d (%.1f%%)\n', overruns, length(chunk_times), ...
        100 * overruns / length(chunk_times));

if avg_chunk_time / chunk_duration_target < 1.0
    fprintf('✓ System capable of real-time operation!\n');
else
    fprintf('⚠ System too slow for real-time!\n');
end

%% Results Analysis
% 1. Time domain signals
figure;
subplot(3,1,1);
plot(t, d_prim, 'b', t, e_mes, 'r');
title('Primary noise vs error comparison');
legend('Primary noise', 'Error after cancellation');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% 2. Before/after spectra
win = hamming(8192*2);
noverlap = 4096*2;
nfft = 8192*2;

[Pxx_d_prim, f] = pwelch(d_prim, win, noverlap, nfft, fs);
[Pxx_e, f] = pwelch(e_mes, win, noverlap, nfft, fs);

subplot(3,1,2);
semilogy(f, Pxx_d_prim, 'b', f, Pxx_e, 'r');
title('Power spectral density');
legend('Primary noise', 'Error after cancellation');
xlabel('Frequency (Hz)');
ylabel('PSD');
xlim([0 500]);
grid on;

% 3. Spectral reduction
reduction_dB = 10*log10(Pxx_e ./ Pxx_d_prim);
subplot(3,1,3);
plot(f, reduction_dB);
title('Acoustic reduction');
xlabel('Frequency (Hz)');
ylabel('Reduction (dB)');
xlim([0 500]);
ylim([-50, 10]);
grid on;

% 4. Adaptive filter coefficients evolution
figure;
subplot(2,1,1);
plot(W);
title('Final adaptive filter coefficients');
xlabel('Coefficient index');
ylabel('Value');
grid on;

% 5. Real-time performance per chunk
subplot(2,1,2);
chunk_numbers = 1:length(chunk_times);
chunk_times_ms = chunk_times * 1000;
chunk_duration_target_line = chunk_duration_target * ones(size(chunk_times));

plot(chunk_numbers, chunk_times_ms, 'b-', 'LineWidth', 1);
hold on;
plot(chunk_numbers, chunk_duration_target_line, 'r--', 'LineWidth', 2);
fill([chunk_numbers, fliplr(chunk_numbers)], ...
     [chunk_times_ms', fliplr(chunk_duration_target_line')], ...
     'red', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

title('Real-Time Performance per Chunk');
xlabel('Chunk Number');
ylabel('Processing time (ms)');
legend('Processing time', 'Real-time limit', 'Overrun zone', 'Location', 'best');
grid on;
ylim([0, max(max(chunk_times_ms), chunk_duration_target) * 1.2]);

%% Performance analysis in target band
band_idx = find(f >= f_low & f <= f_high);
avg_reduction_band = mean(reduction_dB(band_idx));
fprintf('Average reduction in %d-%d Hz band: %.2f dB\n', f_low, f_high, avg_reduction_band);

%% Audio signal export
% Signal normalization for export to avoid clipping
max_val = max(max(abs(d_prim)), max(abs(e_mes)));
d_prim_norm = d_prim / (max_val + 1e-5) * 0.9;
e_mes_norm = e_mes / (max_val + 1e-5) * 0.9;

% Output filename definition
output_filename_before = 'noise_before_anc.wav';
output_filename_after = 'noise_after_anc.wav';

% Save .wav files
audiowrite(output_filename_before, d_prim_norm, fs);
audiowrite(output_filename_after, e_mes_norm, fs);

fprintf('\nAudio files exported:\n');
fprintf('- Noise before ANC: %s\n', output_filename_before);
fprintf('- Noise after ANC: %s\n', output_filename_after);