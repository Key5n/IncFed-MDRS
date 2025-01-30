# SMD
git clone https://github.com/NetManAIOps/OmniAnomaly.git
mv OmniAnomaly/ServerMachineDataset datasets/
rm -rf OmniAnomaly/

# PSM
mkdir -p datasets/PSM
git clone https://github.com/eBay/RANSynCoders.git
mv RANSynCoders/data/* datasets/PSM/
rm -rf RANSynCoders/

# SMAP
# curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
# curl -O https://raw.githubusercontent.com/khundman/telemanom/refs/heads/master/labeled_anomalies.csv
pip install kaggle
# make sure you have an Kaggle API key setup, then:
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl && mv nasa-anomaly-detection-dataset-smap-msl.zip data.zip && unzip -o data.zip && rm data.zip && mv data/data tmp && rm -r data && mv tmp data
