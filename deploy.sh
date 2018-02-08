hexo generate
cp -R public/* .deploy/huangyedi2012.github.io
cd .deploy/huangyedi2012.github.io
git add .
git commit -m update
git push origin master