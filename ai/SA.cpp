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
const int INNER_LOOP = 40;
const int DFS_INITIAL_SEARCH_LIMIT = 5000000;
const int DFS_EXTEND_SEARCH_LIMIT = 80;
const int MONSTER_MOVE_STEPS = 2;
const int EARLY_PATH_WINDOW = 6;
const double NOISE_WEIGHT = 2.0;
const double MOVE_DISTANCE_WEIGHT = 3.0;
const double END_DISTANCE_WEIGHT = 0.5;
const double EARLY_JUNCTION_REWARD = 2.0;
const double LATE_JUNCTION_REWARD = 0.5;
const double JUNCTION_REWARD_CAP = 6.0;
const double BACKTRACK_PENALTY = 10.0;
const string MAP_FILE_PATH = "../map/generated_map.txt";

int xh,yh;
int xm,ym;
bool mp[MAP_SIZE][MAP_SIZE];//true if unavailable, false if available, read from file
double cellNoise[MAP_SIZE][MAP_SIZE];
bool hasPreviousMove=false;

struct Point
{
    int x;
    int y;
};

Point previousMoveFrom;
Point previousMoveTo;

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

int torusDelta(int from,int to)
{
    int delta=to-from;
    if(delta>MAP_SIZE/2)delta-=MAP_SIZE;
    if(delta<-MAP_SIZE/2)delta+=MAP_SIZE;
    return delta;
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

int availableNeighborCount(const Point& p)
{
    int count=0;
    for(const Point& nxt:nextPoints(p))
    {
        if(available(nxt))count++;
    }
    return count;
}

Point actualMovePoint(const vector<Point>& path)
{
    int moveStep=min(MONSTER_MOVE_STEPS,(int)path.size()-1);
    return path[moveStep];
}

void initializeCellNoise()
{
    uniform_real_distribution<double> noisePick(-1.0,1.0);
    for(int i=0;i<MAP_SIZE;i++)
    {
        for(int j=0;j<MAP_SIZE;j++)
        {
            cellNoise[i][j]=noisePick(rng);
        }
    }
}

double randomNoiseCost(const vector<Point>& path)
{
    int count=min((int)path.size(),EARLY_PATH_WINDOW);
    if(count<=0)return 0.0;

    double total=0.0;
    for(int i=0;i<count;i++)
    {
        const Point& p=path[i];
        total+=cellNoise[p.x][p.y];
    }
    return NOISE_WEIGHT*(total/(double)count);
}

double chaseCost(const vector<Point>& path)
{
    Point target={xh,yh};
    Point moved=actualMovePoint(path);
    return (double)path.size()
        + MOVE_DISTANCE_WEIGHT*(double)torusDistance(moved,target)
        + END_DISTANCE_WEIGHT*(double)torusDistance(path.back(),target);
}

double junctionReward(const vector<Point>& path)
{
    double junctionReward=0.0;

    for(int i=1;i<(int)path.size();i++)
    {
        int degree=availableNeighborCount(path[i]);
        bool early=i<EARLY_PATH_WINDOW;
        if(degree>=3)
        {
            junctionReward+=early?EARLY_JUNCTION_REWARD:LATE_JUNCTION_REWARD;
        }
    }

    return min(junctionReward,JUNCTION_REWARD_CAP);
}

double previousMovePenalty(const vector<Point>& path)
{
    if(!hasPreviousMove||path.empty())return 0.0;

    Point moved=actualMovePoint(path);
    double penalty=0.0;

    if(samePoint(moved,previousMoveFrom))
    {
        penalty+=BACKTRACK_PENALTY;
    }

    return penalty;
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
    double score=chaseCost(path);
    score+=randomNoiseCost(path);
    score-=junctionReward(path);
    score+=previousMovePenalty(path);
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
    double temperature=100.0;
    uniform_real_distribution<double> realPick(0.0,1.0);

    while(temperature>0.1)
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
        temperature*=0.97;
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

    int cell;
    for(int i=0;i<30;i++)
    {
        for(int j=0;j<30;j++)
        {
            fin>>cell;
            mp[i][j]=(cell!=0);
        }
    }

    fin>>xh>>yh;//human
    fin>>xm>>ym;//monster
    int previousMoveFlag=0;
    if(fin>>previousMoveFlag)
    {
        if(previousMoveFlag==1&&
           fin>>previousMoveFrom.x>>previousMoveFrom.y
              >>previousMoveTo.x>>previousMoveTo.y)
        {
            previousMoveFrom.x=wrap(previousMoveFrom.x);
            previousMoveFrom.y=wrap(previousMoveFrom.y);
            previousMoveTo.x=wrap(previousMoveTo.x);
            previousMoveTo.y=wrap(previousMoveTo.y);
            hasPreviousMove=true;
        }
    }

    initializeCellNoise();

    vector<Point> dfsPath=makeInitialPath();
    // cout<<"DFS result:"<<endl;
    // printMap(dfsPath);

    vector<Point> path=simulatedAnnealing(dfsPath);
    // cout<<endl<<"SA result:"<<endl;
    // printMap(path);

    Point movedMonster=actualMovePoint(path);

    ofstream fout(MAP_FILE_PATH);
    if(!fout)
    {
        cerr<<"Cannot open output file. Please set MAP_FILE_PATH in SA.cpp."<<endl;
        return 1;
    }

    for(int i=0;i<30;i++)
    {
        for(int j=0;j<30;j++)
        {
            fout<<mp[i][j]<<" ";
        }
        fout<<endl;
    }
    fout<<xh<<" "<<yh<<endl;
    fout<<movedMonster.x<<" "<<movedMonster.y;

    return 0;
}
