## GitHub Copilot Chat

- Extension Version: 0.22.4 (prod)
- VS Code: vscode/1.95.1
- OS: Mac

## Network

User Settings:
```json
  "github.copilot.advanced": {
    "debug.useElectronFetcher": true,
    "debug.useNodeFetcher": false
  }
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 140.82.114.6 (103 ms)
- DNS ipv6 Lookup: ::ffff:140.82.114.6 (1 ms)
- Electron Fetcher (configured): HTTP 200 (98 ms)
- Node Fetcher: HTTP 200 (101 ms)
- Helix Fetcher: HTTP 200 (164 ms)

Connecting to https://api.individual.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.114.21 (108 ms)
- DNS ipv6 Lookup: ::ffff:140.82.114.21 (3 ms)
- Electron Fetcher (configured): HTTP 200 (32 ms)
- Node Fetcher: HTTP 200 (114 ms)
- Helix Fetcher: HTTP 200 (104 ms)

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).