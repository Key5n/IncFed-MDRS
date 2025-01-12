# SMD
git clone https://github.com/NetManAIOps/OmniAnomaly.git
mv OmniAnomaly/ServerMachineDataset datasets/
rm -rf OmniAnomaly/

# SMAP
# curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
# curl -O https://raw.githubusercontent.com/khundman/telemanom/refs/heads/master/labeled_anomalies.csv

# PSM
mkdir -p datasets/psm
git clone https://github.com/eBay/RANSynCoders.git
mv RANSynCoders/data datasets/psm/
rm -rf RANSynCoders
