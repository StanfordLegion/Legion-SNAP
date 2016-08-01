/* Copyright 2016 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __OUTER_H__
#define __OUTER_H__

#include "legion.h"

using namespace Legion;

class CalcOuterSource : public IndexLauncher {
public:
  CalcOuterSource(const Predicate &pred);
public:
  void dispatch(Context ctx, Runtime *runtime);
};

class TestOuterConvergence : public IndexLauncher {
public:
  TestOuterConvergence(const Predicate &pred);
public:
  Future dispatch(Context ctx, Runtime *runtime);
};

#endif // __OUTER_H__

