#pragma once
#include <crow.h>
#include <string>
#include</home/api_projects/include/handlers/interface.hpp>
#include </home/api_projects/include/handlers/users.hpp>
#include <memory>
#include <vector>



struct ServerConfig{
    int port=8080;
    int threads=2;
    std::string log_level="info";
    bool corse =true;
    std::string crossOrgin ="*";

};
namespace servernamespace
{

class Server
{

public:

    explicit Server(const ServerConfig &config=ServerConfig());
    void start();
private:
    ServerConfig config_;
    std::unique_ptr<MyApp> app_;
    std::vector<std::shared_ptr <IHandler>> handler_;
    void setup();
    void addHandler(std::shared_ptr<IHandler> handler);

};
}