FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt update && \
    apt install -y wget curl git build-essential software-properties-common

RUN wget https://github.com/neovim/neovim/releases/download/v0.9.5/nvim-linux64.tar.gz && \
    tar -zxvf nvim-linux64.tar.gz && \
    mv nvim-linux64/bin/nvim /usr/bin/nvim && \
    mv nvim-linux64/lib/nvim /usr/lib/nvim && \
    mv nvim-linux64/share/nvim/ /usr/share/nvim && \
    rm -rf nvim-linux64 && \
    rm nvim-linux64.tar.gz && \
    echo 'alias vi="nvim"' >> ~/.bashrc

RUN git clone https://github.com/Key5n/dotfiles-nix && \
    rm -rf ~/.config/ && \
    mkdir -p ~/.config/nvim && \
    mv dotfiles-nix/home/base/neovim/nvim ~/.config && \
    rm -rf dotfiles-nix

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt install -y nodejs && \
    npm install -g \
    pyright \
    # dockerfile language server
    dockerfile-language-server-nodejs \
    # toml language server
    @taplo/cli && \
    # lazygit
    LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*') && \
    curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz" && \
    tar xf lazygit.tar.gz lazygit && \
    install lazygit /usr/local/bin && \
    rm -f lazygit.tar.gz lazygit && \
    # lazydocker
    curl https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash
