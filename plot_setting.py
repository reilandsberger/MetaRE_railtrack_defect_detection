import matplotlib

matplotlib.rcParams['figure.dpi'] = 900
matplotlib.rcParams.update({'font.size': 7})
matplotlib.rcParams['font.family'] = 'Arial'

matplotlib.rcParams.update({'axes.linewidth': 0.5})
matplotlib.rcParams.update({'ytick.major.width': 0.5})
matplotlib.rcParams.update({'xtick.major.width': 0.5})
matplotlib.rcParams.update({'ytick.minor.width': 0.5})
matplotlib.rcParams.update({'xtick.minor.width': 0.5})

matplotlib.rcParams.update({'xtick.direction': 'in'})
matplotlib.rcParams.update({'ytick.direction': 'in'})

matplotlib.rcParams.update({'savefig.pad_inches': 0.05}) #default 0.1
matplotlib.rcParams.update({'savefig.format': 'png'}) #default 0.1


matplotlib.rcParams.update({'savefig.transparent': True})
matplotlib.rcParams.update({'figure.subplot.bottom': 0.03,
                            'figure.subplot.hspace': 0,
                            'figure.subplot.left': 0.03,
                            'figure.subplot.right': 0.97,
                            'figure.subplot.top': 0.97,
                            'figure.subplot.wspace': 0}) #default 0.125, 0.2, 0.125, 0.9, 0.88, 0.2