// Copyright 2019 srabiee@cs.umass.edu
// College of Information and Computer Sciences,
// University of Massachusetts Amherst
//
//
// This software is free: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License Version 3,
// as published by the Free Software Foundation.
//
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// Version 3 in the file COPYING that came with this distribution.
// If not, see <http://www.gnu.org/licenses/>.
// ========================================================================

#include "io_access.h"

namespace ORB_SLAM2 {

using std::vector;
using std::string;
using std::cout;
using std::endl;

void RemoveDirectory(const string& path) {
  boost::filesystem::path dir(path.c_str());
  boost::filesystem::remove_all(dir);
}

bool CreateDirectory(const string& path) {
  boost::filesystem::path dir(path.c_str());

  if(!(boost::filesystem::exists(dir))) {
    if (!boost::filesystem::create_directory(dir)){
      return false;
    }
  }
  return true;
}

bool WriteJsonToFile(const string& path,
                     const string& file_name,
                     const Json::Value& json_obj) {
  if(!CreateDirectory(path)) {
    return false;
  }

  std::ofstream file_out;
  file_out.open ((path + file_name).c_str(),
                            std::ios::out | std::ios::trunc);
  file_out << json_obj;
  file_out.close();
  return true;
}

PFM::PFM() {
}

} // namespace ORB_SLAM2

