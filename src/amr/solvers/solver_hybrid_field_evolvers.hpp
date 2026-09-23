#ifndef PHARE_AMR_SOLVERS_SOLVER_HYBRID_FIELD_EVOLVERS_HPP
#define PHARE_AMR_SOLVERS_SOLVER_HYBRID_FIELD_EVOLVERS_HPP

#include "core/numerics/ohm/ohm.hpp"
#include "solver_field_evolvers.hpp"

#include <algorithm>
#include <iostream>

namespace PHARE::solver
{


template<typename Model>
class OhmLevelTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using info_type  = core::OhmInfo;
    using core_type  = core::Ohm<GridLayout>;

public:
    explicit OhmLevelTransformer(info_type const& info, level_t& level, Model& model)
        : info_{info}
        , level_{level}
        , model_{model}
    {
    }

    void operator()(GridLayout& layout, auto&&... args) { core_type{info_, layout}(args...); }

    void operator()(auto& B, auto& J, auto& E, auto& electrons)
    {
        auto& rm = *model_.resourcesManager;
        for (auto& patch : rm.enumerate(level_, electrons, B, J, E))
        {
            auto layout = amr::layoutFromPatch<GridLayout>(*patch);
            auto& n     = electrons.density();
            auto& Ve    = electrons.velocity();
            auto& Pe    = electrons.pressure();

            {
                auto [nMin, nMax]   = std::minmax_element(n.begin(), n.end());
                auto [peMin, peMax] = std::minmax_element(Pe.begin(), Pe.end());
                auto [veMin, veMax] = std::minmax_element(Ve(core::Component::X).begin(),
                                                          Ve(core::Component::X).end());
                auto [bMin, bMax]   = std::minmax_element(B(core::Component::X).begin(),
                                                          B(core::Component::X).end());
                auto [jMin, jMax]   = std::minmax_element(J(core::Component::X).begin(),
                                                          J(core::Component::X).end());
                std::cerr << "DEBUG ohm inputs: n[" << *nMin << "," << *nMax << "] Pe[" << *peMin
                          << "," << *peMax << "] Vex[" << *veMin << "," << *veMax << "] Bx["
                          << *bMin << "," << *bMax << "] Jx[" << *jMin << "," << *jMax << "]"
                          << std::endl;
            }

            (*this)(layout, n, Ve, Pe, B, J, E);

            {
                auto [exMin, exMax] = std::minmax_element(E(core::Component::X).begin(),
                                                          E(core::Component::X).end());
                std::cerr << "DEBUG ohm E: Ex[" << *exMin << "," << *exMax << "]" << std::endl;
            }
        }
    }

    void operator()(auto& B, auto& E, auto& electrons) { (*this)(B, model_.state.J, E, electrons); }

    info_type info_;
    level_t& level_;
    Model& model_;
};

template<typename Model>
OhmLevelTransformer(core::OhmInfo, typename Model::amr_types::level_t&, Model&)
    -> OhmLevelTransformer<Model>;


} // namespace PHARE::solver


#endif /* PHARE_AMR_SOLVERS_SOLVER_HYBRID_FIELD_EVOLVERS_HPP */
