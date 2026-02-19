module.exports = {
  apps: [
    {
      name: 'ai-shorts-studio',
      script: 'python',
      args: 'app.py',
      cwd: '/home/user/webapp',
      env: {
        PORT: 5000,
      },
      watch: false,
      instances: 1,
      exec_mode: 'fork',
      max_restarts: 3,
    }
  ]
}
