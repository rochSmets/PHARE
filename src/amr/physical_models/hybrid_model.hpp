#ifndef PHARE_HYBRID_MODEL_HPP
#define PHARE_HYBRID_MODEL_HPP


#include "core/def.hpp"
#include "core/models/hybrid_state.hpp"
#include "core/data/ions/particle_initializers/particle_initializer_factory.hpp"

#include "initializer/data_provider.hpp"

#include "amr/physical_models/physical_model.hpp"
#include "amr/messengers/hybrid_messenger_info.hpp"
#include "amr/resources_manager/resources_manager.hpp"
#include "amr/data/field/refine/field_refine_operator.hpp"
#include "amr/data/field/refine/field_refiner.hpp"
#include "amr/data/field/field_variable_fill_pattern.hpp"
#include "amr/data/tensorfield/tensor_field_data.hpp"

#include <SAMRAI/xfer/RefineAlgorithm.h>
#include <SAMRAI/xfer/RefineSchedule.h>
#include <SAMRAI/hier/PatchHierarchy.h>

#include <map>
#include <string>

namespace PHARE::solver
{
/**
 * @brief The HybridModel class is a concrete implementation of a IPhysicalModel. The class
 * holds a HybridState and a ResourcesManager.
 */
template<typename GridLayoutT, typename Electromag, typename Ions, typename Electrons,
         typename AMR_Types, typename Grid_t>
class HybridModel : public IPhysicalModel<AMR_Types>
{
public:
    static constexpr auto dimension    = GridLayoutT::dimension;
    static constexpr auto interp_order = GridLayoutT::options.interp_order;

    using Interface              = IPhysicalModel<AMR_Types>;
    using amr_types              = AMR_Types;
    using electrons_t            = Electrons;
    using patch_t                = AMR_Types::patch_t;
    using level_t                = AMR_Types::level_t;
    using physical_quantity_type = core::HybridQuantity;
    using gridlayout_type        = GridLayoutT;
    using electromag_type        = Electromag;
    using vecfield_type          = Electromag::vecfield_type;
    using field_type             = vecfield_type::field_type;
    using grid_type              = Grid_t;
    using ions_type              = Ions;
    using particle_array_type    = Ions::particle_array_type;
    using resources_manager_type = amr::ResourcesManager<gridlayout_type, grid_type>;
    using ParticleInitializerFactory
        = core::ParticleInitializerFactory<particle_array_type, gridlayout_type>;

    static constexpr std::string_view model_type_name = "HybridModel";
    static inline std::string const model_name{model_type_name};


    core::HybridState<Electromag, Ions, Electrons> state;
    std::shared_ptr<resources_manager_type> resourcesManager;


    void initialize(level_t& level) override;


    /**
     * @brief allocate uses the ResourcesManager to allocate HybridState physical quantities on
     * the given Patch at the given allocateTime
     */
    virtual void allocate(patch_t& patch, double const allocateTime) override
    {
        resourcesManager->allocate(state, patch, allocateTime);
    }




    /**
     * @brief fillMessengerInfo describes which variables of the model are to be initialized or
     * filled at ghost nodes.
     */
    void fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& info) const override;


    NO_DISCARD auto setOnPatch(patch_t& patch)
    {
        return resourcesManager->setOnPatch(patch, *this);
    }


    // Ve_/Pe_ are never ghost-communicated between patches (unlike B/E/J/ion
    // moments in fillMessengerInfo below), even though the pressure closure's
    // derivOnSameCentering stencils read neighbor cells. Must be called once
    // per level, after every patch's bulk velocity is computed and before any
    // patch computes its pressure.
    //
    // Needs a real refine operator (not nullptr): on level > 0 a patch's ghost
    // region can reach into territory only covered by the coarser level,
    // which needs genuine interpolation, not a same-resolution copy. Needs a
    // non-overwrite-interior fill pattern so the schedule only touches ghost
    // cells, not the domain values just computed on this patch.
    void fillElectronMomentGhosts(int const levelNumber, double const fillTime)
    {
        if (!hierarchy_)
            throw std::runtime_error(
                "Error - HybridModel::fillElectronMomentGhosts called before setHierarchy()");

        if (not electronGhostAlgoDeclared_)
        {
            auto&& [ve_id, pe_id] = resourcesManager->getIDsList(state.electrons.velocity().name(),
                                                                  state.electrons.pressure().name());
            veGhostAlgo_.algo_->registerRefine(ve_id, ve_id, ve_id, vecFieldRefineOp_,
                                               nonOverwriteInteriorTFfillPattern_);
            peGhostAlgo_.algo_->registerRefine(pe_id, pe_id, pe_id, fieldRefineOp_,
                                               nonOverwriteInteriorFieldFillPattern_);
            electronGhostAlgoDeclared_ = true;
        }

        veGhostAlgo_.getOrCreateSchedule(hierarchy_, levelNumber).fillData(fillTime);
        peGhostAlgo_.getOrCreateSchedule(hierarchy_, levelNumber).fillData(fillTime);
    }

    // A brand new level never gets Pe_'s domain cells set at all: unlike Ve_
    // (recomputed statelessly from ions/J every step, so it self-heals), Pe_
    // carries genuinely evolved state, like B/E. oldLevel is null when there
    // is no same-level source yet (initial refinement), or the replaced level
    // on regrid, whose overlap should be preserved rather than
    // re-interpolated from the coarser level.
    void initElectronPressureOnNewLevel(
        int const levelNumber, double const fillTime,
        std::shared_ptr<SAMRAI::hier::PatchLevel> const& oldLevel = nullptr)
    {
        if (!hierarchy_)
            throw std::runtime_error(
                "Error - HybridModel::initElectronPressureOnNewLevel called before setHierarchy()");

        auto&& [pe_id] = resourcesManager->getIDsList(state.electrons.pressure().name());

        SAMRAI::xfer::RefineAlgorithm initAlgo;
        initAlgo.registerRefine(pe_id, pe_id, pe_id, fieldRefineOp_);

        auto const level = hierarchy_->getPatchLevel(levelNumber);
        initAlgo.createSchedule(level, oldLevel, levelNumber - 1, hierarchy_)
            ->fillData(fillTime);
    }

    // SolverPPC::advanceLevel only gets a bare hierarchy reference; only
    // HybridLevelInitializer::initialize() has the actual shared_ptr that
    // SAMRAI's coarse-fine createSchedule() needs, so it's cached here.
    void setHierarchy(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy)
    {
        hierarchy_ = hierarchy;
    }


    HybridModel(PHARE::initializer::PHAREDict const& dict,
                std::shared_ptr<resources_manager_type> const& _resourcesManager)
        : IPhysicalModel<AMR_Types>{model_name}
        , state{dict}
        , resourcesManager{_resourcesManager}
    {
    }


    virtual ~HybridModel() override {}

    //-------------------------------------------------------------------------
    //                  start the ResourcesUser interface
    //-------------------------------------------------------------------------

    NO_DISCARD bool isUsable() const { return state.isUsable(); }

    NO_DISCARD bool isSettable() const { return state.isSettable(); }

    NO_DISCARD auto getCompileTimeResourcesViewList() const { return std::forward_as_tuple(state); }

    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::forward_as_tuple(state); }

    //-------------------------------------------------------------------------
    //                  ends the ResourcesUser interface
    //-------------------------------------------------------------------------

    std::unordered_map<std::string, std::shared_ptr<core::NdArrayVector<dimension, int>>> tags;

private:
    // schedules are cached by level number, but a regrid replaces the
    // PatchLevel object for that number -- the weak_ptr detects that so a
    // stale schedule isn't silently reused against a destroyed level.
    struct GhostFillAlgo
    {
        auto& getOrCreateSchedule(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                                  int const ilvl)
        {
            auto const level   = hierarchy->getPatchLevel(ilvl);
            auto schedule_iter = schedules_.find(ilvl);
            auto const create_schedule
                = schedule_iter == schedules_.end() or schedule_iter->second.level.lock() != level;

            if (create_schedule)
                schedule_iter
                    = schedules_
                          .insert_or_assign(
                              ilvl, Entry{level, algo_->createSchedule(level, ilvl - 1, hierarchy)})
                          .first;

            return *schedule_iter->second.schedule;
        }

        struct Entry
        {
            std::weak_ptr<SAMRAI::hier::PatchLevel> level; // invalidated if the Level is destroyed
            std::shared_ptr<SAMRAI::xfer::RefineSchedule> schedule;
        };

        std::unique_ptr<SAMRAI::xfer::RefineAlgorithm> algo_
            = std::make_unique<SAMRAI::xfer::RefineAlgorithm>();
        std::map<int, Entry> schedules_;
    };

    using VectorFieldData_t = amr::TensorFieldData<1, GridLayoutT, Grid_t, core::HybridQuantity>;

    using DefaultFieldRefineOp_t
        = amr::FieldRefineOperator<GridLayoutT, Grid_t, amr::DefaultFieldRefiner<dimension>>;
    using DefaultVecFieldRefineOp_t
        = amr::VecFieldRefineOperator<VectorFieldData_t, amr::DefaultFieldRefiner<dimension>>;

    std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy_;

    bool electronGhostAlgoDeclared_ = false;
    GhostFillAlgo veGhostAlgo_;
    GhostFillAlgo peGhostAlgo_;

    std::shared_ptr<SAMRAI::hier::RefineOperator> fieldRefineOp_
        = std::make_shared<DefaultFieldRefineOp_t>();
    std::shared_ptr<SAMRAI::hier::RefineOperator> vecFieldRefineOp_
        = std::make_shared<DefaultVecFieldRefineOp_t>();

    std::shared_ptr<amr::FieldFillPattern<dimension>> nonOverwriteInteriorFieldFillPattern_
        = std::make_shared<amr::FieldFillPattern<dimension>>();
    std::shared_ptr<amr::TensorFieldFillPattern<dimension>> nonOverwriteInteriorTFfillPattern_
        = std::make_shared<amr::TensorFieldFillPattern<dimension>>();
};




//-------------------------------------------------------------------------
//                             definitions
//-------------------------------------------------------------------------


template<typename GridLayoutT, typename Electromag, typename Ions, typename Electrons,
         typename AMR_Types, typename Grid_t>
void HybridModel<GridLayoutT, Electromag, Ions, Electrons, AMR_Types, Grid_t>::initialize(
    level_t& level)
{
    auto& rm = *this->resourcesManager;
    for (auto& patch : rm.enumerate(level, state))
    {
        auto const layout = amr::layoutFromPatch<gridlayout_type>(*patch);

        for (auto& pop : state.ions)
            ParticleInitializerFactory::create(pop.particleInitializerInfo())
                ->loadParticles(pop.domainParticles(), layout);


        state.electrons.initialize(layout);
        state.electromag.initialize(layout);
    }
}



template<typename GridLayoutT, typename Electromag, typename Ions, typename Electrons,
         typename AMR_Types, typename Grid_t>
void HybridModel<GridLayoutT, Electromag, Ions, Electrons, AMR_Types, Grid_t>::fillMessengerInfo(
    std::unique_ptr<amr::IMessengerInfo> const& info) const
{
    auto& hybridInfo = dynamic_cast<amr::HybridMessengerInfo&>(*info);

    // only the charge density is registered to the messenger and not the ion mass
    // density. Reason is that mass density is only used to compute the
    // total bulk velocity which is already registered to the messenger
    hybridInfo.modelMagnetic        = state.electromag.B.name();
    hybridInfo.modelElectric        = state.electromag.E.name();
    hybridInfo.modelIonDensity      = state.ions.chargeDensityName();
    hybridInfo.modelIonBulkVelocity = state.ions.velocity().name();
    hybridInfo.modelCurrent         = state.J.name();

    hybridInfo.initElectric.emplace_back(state.electromag.E.name());
    hybridInfo.initMagnetic.emplace_back(state.electromag.B.name());

    hybridInfo.ghostElectric.push_back(hybridInfo.modelElectric);
    hybridInfo.ghostMagnetic.push_back(hybridInfo.modelMagnetic);
    hybridInfo.ghostCurrent.push_back(state.J.name());
    hybridInfo.ghostBulkVelocity.push_back(hybridInfo.modelIonBulkVelocity);

    auto transform_ = [](auto& ions, auto& inserter) {
        std::transform(std::begin(ions), std::end(ions), std::back_inserter(inserter),
                       [](auto const& pop) { return pop.name(); });
    };
    transform_(state.ions, hybridInfo.interiorParticles);
    transform_(state.ions, hybridInfo.levelGhostParticlesOld);
    transform_(state.ions, hybridInfo.levelGhostParticlesNew);
    transform_(state.ions, hybridInfo.patchGhostParticles);

    for (auto const& pop : state.ions)
    {
        hybridInfo.ghostFlux.emplace_back(pop.flux().name());
        hybridInfo.sumBorderFields.emplace_back(pop.particleDensity().name());
        hybridInfo.sumBorderFields.emplace_back(pop.chargeDensity().name());
    }

    hybridInfo.maxBorderFields.emplace_back(state.ions.massDensity().name());
    hybridInfo.maxBorderFields.emplace_back(state.ions.chargeDensity().name());
    hybridInfo.maxBorderVecFields.emplace_back(state.ions.velocity().name());
}




template<typename Model>
auto constexpr is_hybrid_model(Model* m) -> decltype(m->model_type_name, bool())
{
    return Model::model_type_name == "HybridModel";
}

template<typename... Args>
auto constexpr is_hybrid_model(Args...)
{
    return false;
}

template<typename Model>
auto constexpr is_hybrid_model_v = is_hybrid_model(static_cast<Model*>(nullptr));



} // namespace PHARE::solver

#endif
