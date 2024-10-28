# Partial autocorrelation
plot_pacf(df_notimecollumn[column_name].diff().dropna())

plt.savefig('partialautocorrelationp.png')

plot_acf(df_notimecollumn[column_name].diff().dropna())

plt.savefig('partialautocorrelationq.png')

# Partial autocorrelation second derivative
plot_pacf(df_notimecollumn[column_name].diff().diff().dropna())

plt.savefig('partialautocorrelationp2.png')

plot_acf(df_notimecollumn[column_name].diff().diff().dropna())

plt.savefig('partialautocorrelationq2.png')