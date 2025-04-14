#!/bin/bash

git config --global user.email "tuandung12092002@gmail.com"
git config --global user.name "tuandung222"

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "❌ Error: No commit message provided."
  echo "✅ Usage: $0 \"Your commit message here\""
  exit 1
fi

# Assign commit message from parameter
MESSAGE="$1"

# Display current git branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "📂 Working on branch: $BRANCH"

# Git commands
echo "📦 Adding changes..."
git add .

echo "📝 Committing with message: \"$MESSAGE\""
git commit -m "$MESSAGE"

echo "🚀 Pushing to origin/$BRANCH..."
git push origin "$BRANCH"

echo "✅ Done!"
