#ifndef PHARE_TESTS_AMR_TOOLS_RESSOURCE_RESSOURCE_TEST_1D_H
#define PHARE_TESTS_AMR_TOOLS_RESSOURCE_RESSOURCE_TEST_1D_H

#include <memory>


#include "test_resources_manager_basic_hierarchy.h"
#include "core/data/field/field.h"
#include "core/data/grid/gridlayout.h"
#include "core/data/grid/gridlayout_impl.h"
#include "core/data/ions/ion_population/ion_population.h"
#include "core/data/ions/ions.h"
#include "core/data/ndarray/ndarray_vector.h"
#include "core/data/particles/particle_array.h"
#include "core/data/vecfield/vecfield.h"
#include "input_config.h"
#include "amr/resources_manager/resources_manager.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace PHARE::core;
using namespace PHARE::amr;




template<typename ResourcesUsers>
class aResourceUserCollection : public ::testing::Test
{
public:
    std::unique_ptr<BasicHierarchy> hierarchy;
    ResourcesManager<GridLayout<GridLayoutImplYee<1, 1>>> resourcesManager;

    ResourcesUsers users;

    void SetUp()
    {
        auto s    = inputBase + std::string("/input/input_db_1d");
        hierarchy = std::make_unique<BasicHierarchy>(inputBase + std::string("/input/input_db_1d"));
        hierarchy->init();

        auto registerAndAllocate = [this](auto& resourcesUser) {
            auto& patchHierarchy = hierarchy->hierarchy;

            resourcesManager.registerResources(resourcesUser.user);

            double const initDataTime{0.0};

            for (int iLevel = 0; iLevel < patchHierarchy->getNumberOfLevels(); ++iLevel)
            {
                auto patchLevel = patchHierarchy->getPatchLevel(iLevel);
                for (auto& patch : *patchLevel)
                {
                    resourcesManager.allocate(resourcesUser.user, *patch, initDataTime);
                }
            }
        }; // end lambda

        std::apply(registerAndAllocate, users);
    }
};




#endif
