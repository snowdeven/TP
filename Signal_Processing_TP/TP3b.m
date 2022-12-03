clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Butterworth low pass filter of order 3
% First IIR filter stage corresponding to pole exp(j2pi/3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

phik=2*pi/3; % First pole 
FS=44100;  % Sampling frequency
fc=1e3; % Cut-off frequency
alpha=1/tan(pi*fc/FS); % Scaling factor

B=[1 2 1]; % B coefficients
A=[1-2*alpha*cos(phik)+alpha^2 2*(1-alpha^2) (1+2*alpha*cos(phik)+alpha^2)]; % A coefficients 

Nb=5*FS; % Amount of evaluated values 
s=randn(Nb,1); % Random generated number (noise)

df=FS/Nb; % Frequency step
f=(-Nb/2:Nb/2-1)*df; % Interval
H=20*log10(abs(fftshift(fft(s)))); % Test signal 

% Plots
figure(1)
subplot (311) 
plot(f,H) 
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
ylim([-60 80])
title('Power density spectrum of test signal')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply First IIR filter stage corresponding to pole exp(j2pi/3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s2=filter(B,A,s); % Butterworth filter

H2=20*log10(abs(fftshift(fft(s2)))); % 1st IIR stage signal

% Plots
figure(1)
subplot (312) 
plot(f,H2) 
grid on
set(gca,'Xscale','log');
xlim([20 20e3])
ylim([-60 80])
xlabel('frequency (Hz)')
ylabel('H (dB)')
title('Power density spectrum after 1st stage IIR filter')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Butterworth low pass filter of order 3
% SECOND IIR filter stage (bloc) corresponding to pole exp(j.pi)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B2=[1 1]; % B coeffs
A2=[1+FS/(pi*fc) 1-FS/(pi*fc)]; % A coeffs

s3=filter(B2,A2,s2); % 2nd IIR stage signal

H3=20*log10(abs(fftshift(fft(s3)))); % 2nd signal

% Plot of full butterworth
figure(1)
subplot (313)
plot(f,H3)
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
ylim([-60 80])
xlabel('frequency (Hz)')
ylabel('H (dB)')
title('Power density spectrum after 2nd stage IIR filter (Full butterworth)')

set(gcf,'Position',[96   105   547   658]); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Exercice %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[Y,FS2]=audioread('Adele.wav');
[Y,FS2]=audioread('Adele.wav');

% Setting parameters just as in the first part
NbF=2^14;
t=(1:NbF)/FS2; 
Nb2=length(Y);
df2=FS2/Nb2; 
f2=(-Nb2/2:Nb2/2-1)*df2; 

% Original signal plot
H_original=20*log10(abs(fftshift(fft(Y(:,1))))); 
figure(2)
subplot (311)
plot(f2,H_original(:,1)) 
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
ylim([-60 80])
title('Adele original')

fil=filter(B,A,Y); % Creating the 1st stage filter with respect to the signal Y
H_fil=20*log10(abs(fftshift(fft(fil)))); % Signal

% Plotting the 1st order for fc=1000Hz
figure(2)
subplot (312)
plot(f2,H_fil(:,1)) 
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
ylim([-60 80])
title('Adele order 1, fc=1 000Hz')

fil2=filter(B2,A2,fil); % Creating the 2nd stage filter with respect to the signal Y
H_fil2=20*log10(abs(fftshift(fft(fil2)))); % Signal

% Plotting the 2nd order for fc=1000Hz
figure(2)
subplot (313)
plot(f2,H_fil2(:,1)) 
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
ylim([-60 80])
title('Adele order 2, , fc=1 000Hz')

%soundsc(Y,FS2)
%pause(20)
%soundsc(fil,FS) 
%pause(20)
%soundsc(fil2,FS) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% different cutoff freq%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Same steps as above but for fc = 100
fc=1e2; 
alpha=1/tan(pi*fc/FS); 

B=[1 2 1]; 
A=[1-2*alpha*cos(phik)+alpha^2 2*(1-alpha^2) (1+2*alpha*cos(phik)+alpha^2)]; 


fil=filter(B,A,Y); 
H_fil=20*log10(abs(fftshift(fft(fil)))); % 1st stage
fil2=filter(B2,A2,fil);
H_fil2=20*log10(abs(fftshift(fft(fil2)))); % 2nd stage

figure(2)
subplot (311)
plot(f2,H_fil2(:,1)) 
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
ylim([-60 80])
title('Adele order 2, fc=1 00Hz')

% Same steps as above but for fc = 1000
fc=1e3; 
alpha=1/tan(pi*fc/FS);

B=[1 2 1]; 
A=[1-2*alpha*cos(phik)+alpha^2 2*(1-alpha^2) (1+2*alpha*cos(phik)+alpha^2)]; 


fil=filter(B,A,Y); 
H_fil=20*log10(abs(fftshift(fft(fil)))); % 1st stage
fil2=filter(B2,A2,fil);
H_fil2=20*log10(abs(fftshift(fft(fil2)))); % 2nd stage

figure(2)
subplot (312)
plot(f2,H_fil2(:,1)) 
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
ylim([-60 80])
title('Adele order 1, fc=1 000Hz')

% Same steps as above but for fc = 5000
fc=5e3; 
alpha=1/tan(pi*fc/FS); 

B=[1 2 1]; 
A=[1-2*alpha*cos(phik)+alpha^2 2*(1-alpha^2) (1+2*alpha*cos(phik)+alpha^2)]; 


fil=filter(B,A,Y); 
H_fil=20*log10(abs(fftshift(fft(fil)))); % 1st stage 
fil2=filter(B2,A2,fil);
H_fil2=20*log10(abs(fftshift(fft(fil2)))); % 2nd stage

figure(2)
subplot (313)
plot(f2,H_fil2(:,1)) 
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
ylim([-60 80])
title('Adele order 2, fc=5 000Hz')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% low pass filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Wn=fc/(FS/2); % Normalized cut-off frequency 

[b,a]=butter(3,Wn,'LOW'); % Getting the coefficients from Butterworth
low1=filter(b,a,s); % Builds a filter with respect to these coefficients on s
H_low1=20*log10(abs(fftshift(fft(low1)))); % Signal

figure(3)
subplot 211
plot(f,H3(:,1)) % Signal with hand-made coefficients
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
%ylim([-60 80])
title('Filter order 1 and 2, fc=1 000Hz, low pass')

figure(3)
subplot (212)
plot(f,H_low1) % Signal with retrieved coefficients
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
%ylim([-60 80])
title('Filter with butter function, fc=1 000Hz, low pass')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% high pass %%%%%%%%%%ZZ%%%%%%%%%%%%%%%%%%%%%%

Wn2=500/(FS/2); % Normalized cut-off frequency 
Wn3 = [1000/(FS/2) 2000/(FS/2)]; % Setting the band pass 
[b2,a2]=butter(3,Wn2,'HIGH'); % Getting the coefficients from Butterworth (high)
[b3,a3]=butter(3,Wn3,'STOP'); % Getting the coefficients from Butterworth (band pass)

high_adele = filter(b2,a2,Y(:,1)); % Filtered signal for high pass
low_adele = filter(b,a,Y(:,1)); % Filtered signal for low pass
stop_adele = filter(b3,a3,Y(:,1)); % Filtered signal for band pass

H_low_adele=20*log10(abs(fftshift(fft(low_adele)))); % Signal in log scale
H_high_adele=20*log10(abs(fftshift(fft(high_adele))));
H_stop_adele=20*log10(abs(fftshift(fft(stop_adele))));

figure(4)
subplot (411)
plot(f2,H_original) 
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
%ylim([-60 80])
title('Adele original')

figure(4)
subplot (412)
plot(f2,H_high_adele)
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
%ylim([-60 80])
title('Adele high pass filter, fc=1 000Hz')

figure(4)
subplot (413)
plot(f2,H_low_adele)
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
%ylim([-60 80])
title('Adele low pass filter, fc=1 000Hz')

figure(4)
subplot (414)
plot(f2,H_stop_adele)
grid on
set(gca,'Xscale','log'); 
xlim([20 20e3])
xlabel('frequency (Hz)')
ylabel('H (dB)')
%ylim([-60 80])
title('Adele stop band , fc=[1000Hz ; 2000Hz] ')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Listening %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%soundsc(Y(1:797416,1),FS2) % original sound
%pause(45)
%soundsc(high_adele(1:797416,1),FS) % high pass filter
%pause(45)
%soundsc(low_adele(1:797416,1),FS) % low pass filter
%pause(45)
%soundsc(stop_adele(1:797416,1),FS) % stop band filter
