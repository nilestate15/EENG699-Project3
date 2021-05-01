import numpy as np 
import scipy.signal as sig
import matplotlib.pyplot as plt

def C_A_sig(f_0, f_s, t_max):
    ''' Function generates simulated C/A code signals.
    Inputs: f_0 (center frequency), f_s (sampling frequency), t_max (length of sampled data to simulate)
    Ouput: 2 arrays - Carrier only signal & C/A code signal modulated onto the carrier '''
    # C/A chipping rate
    f_chip = 1023000 #chips/sec

    # Time vector generation
    time_array = np.arange(0, t_max, 1/f_s)

    # Calculate chip number
    chip_num = np.floor(time_array * f_chip).astype('int')

    # Carrier frequency
    carr_freq = np.cos(2*np.pi*f_0*time_array)

    # Generate Sampling of PRN codes
    prn_code = np.random.randint(2, size=len(chip_num))
    prn_code[prn_code == 0] = -1

    prn_sampled = prn_code[chip_num]

    # Generate C/A code signal modulated onto the carrier
    mod_freq = carr_freq * prn_sampled

    return carr_freq, mod_freq


def CA_pwr_spec(f_s, t_max, carr_sig, mod_sig):
    '''Function calculates the power spectral density (PSD)
    Inputs: f_s (sampling frequency), t_max (length of sampled data to simulate), carr_sig (carrier signal), mod_sig (C/A code modulated onto the carrier)
    Output: an array of the spectral density (PSD)'''

     # Time vector generation
    time_array = np.arange(0, t_max, 1/f_s)

    # Normalized FFT output of the signals
    fft_signal_carr = np.fft.fft(carr_sig)/len(carr_sig)
    fft_signal_carr = abs(fft_signal_carr[range(int(len(carr_sig)/2))])

    fft_signal_mod = np.fft.fft(mod_sig)/len(mod_sig)
    fft_signal_mod = abs(fft_signal_mod[range(int(len(mod_sig)/2))])

    # Frequency Axis
    len_time = len(time_array)
    values = np.arange(int(len_time/2))
    time_period = len_time / f_s
    freq_axis = values / time_period

    # Filtered FFT output for C/A modulated
    N = 500
    fft_filtered_mod = sig.filtfilt(np.ones(N), 1, fft_signal_mod)

    # Plotting
    # Carrier Signal PSD
    plt.figure()
    plt.title('Spectral Density Carrier Signal')
    plt.plot(freq_axis, fft_signal_carr, marker = '.', markersize = 5)
    plt.legend(['PSD Carrier'])
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Power')
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.show()

    # C/A code Signal PSD
    plt.figure()
    plt.title('Spectral Density C/A code signal Modulated on Carrier')
    plt.plot(freq_axis, fft_filtered_mod, marker = '.', markersize = 5)
    plt.legend(['PSD C/A Modulated'])
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Power')
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.show()


def P_Y_sig(f_0, f_s, t_max):
    ''' Function generates simulated P(Y) code signals.
    Inputs: f_0 (center frequency), f_s (sampling frequency), t_max (length of sampled data to simulate)
    Output: 1 array - P(Y) code signal modulated onto the carrier '''
    # P(Y) chipping rate (10x faster than C/A)
    f_chip = 10230000 #chips/sec

    # Time vector generation
    time_array = np.arange(0, t_max, 1/f_s)

    # Calculate chip number
    chip_num = np.floor(time_array * f_chip).astype('int')

    # Carrier frequency
    carr_freq = np.cos(2*np.pi*f_0*time_array)

    # Generate Sampling of PRN codes
    prn_code = np.random.randint(2, size=len(chip_num))
    prn_code[prn_code == 0] = -1

    prn_sampled = prn_code[chip_num]

    # Generate P(Y) code signal modulated onto the carrier
    mod_freq = carr_freq * prn_sampled

    # L1 P(Y) code has 3dB less power than C/A
    PY_sig = mod_freq * 1/np.sqrt(2)

    return PY_sig



def PY_pwr_spec(f_s, t_max, mod_sig):
    '''Function calculates the power spectral density (PSD)
    Inputs: f_s (sampling frequency), t_max (length of sampled data to simulate), mod_sig (PY code modulated onto the carrier)
    Output: an array of the spectral density (PSD)'''

     # Time vector generation
    time_array = np.arange(0, t_max, 1/f_s)

    # Normalized FFT output of the signals
    fft_signal_mod = np.fft.fft(mod_sig)/len(mod_sig)
    fft_signal_mod = abs(fft_signal_mod[range(int(len(mod_sig)/2))])

    # Frequency Axis
    len_time = len(time_array)
    values = np.arange(int(len_time/2))
    time_period = len_time / f_s
    freq_axis = values / time_period

    # Filtered FFT output for C/A modulated
    N = 500
    fft_filtered_mod = sig.filtfilt(np.ones(N), 1, fft_signal_mod)

    # Plotting Spectral Density P(Y)
    plt.figure()
    plt.title('Spectral Density P(Y) code signal Modulated on Carrier')
    plt.plot(freq_axis, fft_filtered_mod, marker = '.', markersize = 5)
    plt.legend(['PSD P(Y) Modulated'])
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Power')
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.show()


def M_C_A_sig(f_0, f_s, t_max):
    ''' Function generates simulated C/A code signals for M code.
    Inputs: f_0 (center frequency), f_s (sampling frequency), t_max (length of sampled data to simulate)
    Ouput: 1 array - Carrier only signal & C/A code signal modulated onto the carrier '''
    # PRN chipping rate
    f_chip = 5 * 1023000 #chips/sec

    # M-code frequency
    m_freq = 10 * 1023000

    # Time vector generation
    time_array = np.arange(0, t_max, 1/m_freq)

    # Square wave
    square_freq = sig.square(2*np.pi*f_0*time_array)

    # Calculate chip number
    chip_num = np.floor(time_array * f_chip).astype('int')

    # Generate Sampling of PRN codes
    prn_code = np.random.randint(2, size=len(chip_num))
    prn_code[prn_code == 0] = -1

    prn_sampled = prn_code[chip_num]

    # M-code (modulation of prn onto square wave)
    m_code = square_freq * prn_sampled

    return m_code

def M_pwr_spec(m_c, t_max, mod_sig):
    '''Function calculates the power spectral density (PSD)
    Inputs: m_c (M_code freq), t_max (length of sampled data to simulate), mod_sig (PY code modulated onto the carrier)
    Output: an array of the spectral density (PSD)'''

     # Time vector generation
    time_array = np.arange(0, t_max, 1/m_c)

    # Normalized FFT output of the signals
    fft_signal_mod = np.fft.fft(mod_sig)/len(mod_sig)
    fft_signal_mod = abs(fft_signal_mod[range(int(len(mod_sig)/2))])

    # Frequency Axis
    len_time = len(time_array)
    values = np.arange(int(len_time/2))
    time_period = len_time / m_c
    freq_axis = values / time_period

    # Filtered FFT output for M code modulated
    N = 500
    fft_filtered_mod = sig.filtfilt(np.ones(N), 1, fft_signal_mod)

    # Plotting Spectral Density M code
    plt.figure()
    plt.title('Spectral Density M code signal')
    plt.plot(freq_axis, fft_filtered_mod, marker = '.', markersize = 5)
    plt.legend(['PSD M code'])
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Power')
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.show()



if __name__ == "__main__":
    # Settings
    # Center Frequency (30MHz)
    cent_freq = 30000000
    # Sample Frequency (200MHz)
    samp_freq = 200000000
    # Sampled Data Length (0.1secs)
    t_max = 0.1 
    # M Code Frequency (10*1.023MHz)
    m_freq = 10 * 1023000

    ## For P code
    # Generate Simulated C/A Code Signal
    carrier_sig, CA_sig = C_A_sig(cent_freq, samp_freq, t_max)

    # Calculate Power Spectral Density for C/A Code and plot
    # CA_pwr_spec(samp_freq, t_max, carrier_sig, CA_sig)

    # Generate Simulated P(Y) Code Signal
    # PY_sig = P_Y_sig(cent_freq, samp_freq, t_max)

    # Calculate Power Spectral Density for P(Y) Code and plot
    # PY_pwr_spec(samp_freq, t_max, PY_sig)

    ## For M code
    # Generate Simulated C/A Code Signal
    M_sig = M_C_A_sig(cent_freq, samp_freq, t_max)

    # Calculate Power Spectral Density for P(Y) Code and plot
    M_pwr_spec(m_freq, t_max, M_sig)


