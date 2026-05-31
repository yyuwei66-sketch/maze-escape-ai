#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
using namespace std;

const int MAP_SIZE = 30;
const int MAX_PATH_LEN = 100;
const int INNER_LOOP = 200;
const int DFS_INITIAL_SEARCH_LIMIT = 5000000;
const int DFS_EXTEND_SEARCH_LIMIT = 80;
const string MAP_FILE_PATH = "../map/test2.txt";

int xh,yh;
int xm,ym;
bool mp[MAP_SIZE][MAP_SIZE];//true if unavailable, false if available, read from file

struct Point
{
    int x;
    int y;
};

mt19937 rng(random_device{}());

int wrap(int v)
{
    return (v + MAP_SIZE) % MAP_SIZE;
}

bool samePoint(const Point& a,const Point& b)
{
    return a.x==b.x&&a.y==b.y;
}

bool available(const Point& p)
{
    return !mp[p.x][p.y];
}

int torusDistance(const Point& a,const Point& b)
{
    int dx=abs(a.x-b.x);
    int dy=abs(a.y-b.y);
    dx=min(dx,MAP_SIZE-dx);
    dy=min(dy,MAP_SIZE-dy);
    return dx+dy;
}

vector<Point> nextPoints(const Point& p)
{
    vector<Point> ans;
    ans.push_back({wrap(p.x+1),p.y});
    ans.push_back({wrap(p.x-1),p.y});
    ans.push_back({p.x,wrap(p.y+1)});
    ans.push_back({p.x,wrap(p.y-1)});
    return ans;
}

vector<Point> orderedDfsCandidates(const Point& now,const Point& target,bool visited[MAP_SIZE][MAP_SIZE])
{
    vector<Point> candidates;
    for(const Point& p:nextPoints(now))
    {
        if(!available(p)||visited[p.x][p.y])continue;
        candidates.push_back(p);
    }

    if(candidates.empty())return candidates;

    int minDist=torusDistance(candidates[0],target);
    for(const Point& p:candidates)
    {
        minDist=min(minDist,torusDistance(p,target));
    }

    vector<Point> nearest;
    vector<Point> others;
    for(const Point& p:candidates)
    {
        if(torusDistance(p,target)==minDist)nearest.push_back(p);
        else others.push_back(p);
    }

    shuffle(nearest.begin(),nearest.end(),rng);
    shuffle(others.begin(),others.end(),rng);

    nearest.insert(nearest.end(),others.begin(),others.end());
    return nearest;
}

bool dfsInitialPath(const Point& now,const Point& target,bool visited[MAP_SIZE][MAP_SIZE],vector<Point>& path,int& searchLeft)
{
    if(samePoint(now,target))return true;
    if((int)path.size()>=MAX_PATH_LEN)return false;
    if(searchLeft<=0)return false;

    searchLeft--;

    vector<Point> candidates=orderedDfsCandidates(now,target,visited);

    for(const Point& nxt:candidates)
    {
        visited[nxt.x][nxt.y]=true;
        path.push_back(nxt);

        if(dfsInitialPath(nxt,target,visited,path,searchLeft))return true;

        path.pop_back();
        visited[nxt.x][nxt.y]=false;
    }

    return false;
}

vector<Point> extendPath(vector<Point> path,const Point& target)
{
    bool visited[MAP_SIZE][MAP_SIZE]={false};
    vector<Point> originalPath=path;
    int searchLeft=DFS_EXTEND_SEARCH_LIMIT;

    for(const Point& p:path)
    {
        visited[p.x][p.y]=true;
    }

    if(dfsInitialPath(path.back(),target,visited,path,searchLeft))return path;

    return originalPath;
}

vector<Point> makeInitialPath()
{
    vector<Point> path;
    Point start={xm,ym};
    Point target={xh,yh};
    bool visited[MAP_SIZE][MAP_SIZE]={false};

    path.push_back(start);
    visited[start.x][start.y]=true;
    int searchLeft=DFS_INITIAL_SEARCH_LIMIT;

    if(dfsInitialPath(start,target,visited,path,searchLeft))return path;

    return path;
}

double scorePath(const vector<Point>& path)
{
    Point target={xh,yh};
    int repeated[MAP_SIZE][MAP_SIZE]={0};
    int repeatPenalty=0;

    for(const Point& p:path)
    {
        repeated[p.x][p.y]++;
        if(repeated[p.x][p.y]>1)repeatPenalty++;
    }

    int distance=torusDistance(path.back(),target);
    double score=(double)path.size()+repeatPenalty*2.0;

    if(!samePoint(path.back(),target))score+=distance*50.0+1000.0;
    return score;
}

vector<Point> mutatePath(const vector<Point>& path)
{
    if(path.size()<=1)return extendPath(path,{xh,yh});

    uniform_int_distribution<int> cutPick(0,(int)path.size()-1);
    int cut=cutPick(rng);

    vector<Point> nextPath;
    for(int i=0;i<=cut;i++)nextPath.push_back(path[i]);

    return extendPath(nextPath,{xh,yh});
}

vector<Point> simulatedAnnealing(const vector<Point>& initialPath)
{
    vector<Point> current=initialPath;
    vector<Point> best=current;
    double currentScore=scorePath(current);
    double bestScore=currentScore;
    double temperature=1000.0;
    uniform_real_distribution<double> realPick(0.0,1.0);

    while(temperature>1e-3)
    {
        for(int i=0;i<INNER_LOOP;i++)
        {
            vector<Point> candidate=mutatePath(current);
            double candidateScore=scorePath(candidate);
            double delta=currentScore-candidateScore;

            if(delta>0||exp(delta/temperature)>realPick(rng))
            {
                current=candidate;
                currentScore=candidateScore;
 
                if(currentScore<bestScore)
                {
                    best=current;
                    bestScore=currentScore;
                }
            }
        }
        temperature*=0.995;
    }

    return best;
}

void printMap(const vector<Point>& path)
{
    int stepMap[MAP_SIZE][MAP_SIZE];
    for(int i=0;i<MAP_SIZE;i++)
    {
        for(int j=0;j<MAP_SIZE;j++)
        {
            stepMap[i][j]=-1;
        }
    }

    for(int i=0;i<(int)path.size();i++)
    {
        const Point& p=path[i];
        if(stepMap[p.x][p.y]==-1)stepMap[p.x][p.y]=i;
    }

    for(int i=0;i<MAP_SIZE;i++)
    {
        for(int j=0;j<MAP_SIZE;j++)
        {
            if(mp[i][j])cout<<"#";
            else if(i==xh&&j==yh)cout<<"H";
            else if(i==xm&&j==ym)cout<<"M";
            else if(stepMap[i][j]!=-1)cout<<stepMap[i][j]%10;
            else cout<<" ";
        }
        cout<<endl;
    }
}

int main()
{
    ifstream fin(MAP_FILE_PATH);
    if(!fin)
    {
        cerr<<"Cannot open map file. Please set MAP_FILE_PATH in SA.cpp."<<endl;
        return 1;
    }

    int n=0;
    fin>>n;
    for(int i=1;i<=n;i++)
    {
        int x,y;
        fin>>x>>y;
        mp[x][y]=true;
    }
/*
    for(int i=0;i<30;i++)
    {
        for(int j=0;j<30;j++)
        {
            if(mp[i][j])cout<<"*";
            else cout<<" ";
        }
        cout<<endl;
    }
*/
    fin>>xh>>yh;//human
    fin>>xm>>ym;//monster

    vector<Point> dfsPath=makeInitialPath();
    cout<<"DFS result:"<<endl;
    printMap(dfsPath);

    cout<<endl<<"SA result:"<<endl;
    vector<Point> path=simulatedAnnealing(dfsPath);
    printMap(path);

    return 0;
}
