clear all
close all
clc

[signal,FS] = audioread('Windows_Ding.wav');
Nb = length(signal);
time = (0:Nb-1)/FS;
step = (0:Nb-1)/time;

figure(1)
plot(time,signal(:,1))
grid on
xlabel('time (s)')
ylabel('amplitude')
title('Original wav file')
[tt,aa] = ginput(2);
CycleDuration = tt(2) - tt(1);

df = FS/Nb;
frequency = (-Nb/2:Nb/2-1)*df;
spectrum = fft(signal(:,1));
spectrum = fftshift(spectrum);
spectrumdB = 20*log10(abs(spectrum).^2);

figure(2)
plot(frequency,spectrum)
grid on
xlim([0 2000])
xlabel('Frequency (Hz)')
ylabel('Amplitude')
title('Power density original scale')

NbO = 100;
[pks,locs] = findpeaks(abs(spectrum).^2,'sortstr','descend','Npeaks',NbO);
freq_resonance = frequency(locs);

figure(3)
plot(frequency,abs(spectrum))
hold on
plot(frequency(locs),abs(spectrum(locs)),'or')
hold off
grid on
xlim([0 2000])
xlabel('Frequency (Hz)')
ylabel('Amplitude')

synth_signal = 0;
tau = 0.2;
for i=1:NbO,
    synth_signal = synth_signal + spectrum(locs(i)) * exp(j*2*pi*freq_resonance(i)*time).*exp(-time/tau);
end;

synth_signal = real(synth_signal);
synth_signal = synth_signal/max(abs(synth_signal))*0.9;

figure(4)
plot(time,synth_signal)
grid on
xlabel('t (s)')
ylabel('Amplitude')
title('Synthetized audio')

S = fftshift(fft(synth_signal));
S = S/max(abs(S))*70;

figure(5)
plot(frequency,abs(spectrum))
hold on
plot(frequency(locs),abs(spectrum(locs)),'or')
plot(frequency,abs(S),'r')
hold off
grid on
xlim([0 6000])
xlabel('Frequency (Hz)')
ylabel('Amplitude')
title('original spectrum,detected resonances,synthetized spectrum')
legend('original spectrum','detected resonances','synthetized spectrum')




