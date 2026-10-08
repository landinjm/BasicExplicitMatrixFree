#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exception_macros.h>
#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

/**
 * Simple matrix-free & block vector problem.
 *
 * We simultaneously solve duplicates of the same problem. For simplicity, we
 * just add one each increment (i.e., x^n = 1 + x^n-1).
 */

constexpr int dim = 2;
constexpr int degree = 4;
constexpr int n_refinements = 3;
constexpr int n_steps = 10;
constexpr int n_blocks = 21;
using RealType = double;

using namespace dealii;

/**
 * Simple operator that adds one.
 *
 * @note No AMR allowed so const inverted mass matrix
 */
class Operator {
public:
  using MF = MatrixFree<dim, RealType>;
  using Vector = LinearAlgebra::distributed::BlockVector<RealType>;
  using Vector2 = LinearAlgebra::distributed::Vector<RealType>;

  Operator(const MF &data) : _data(data) {
    _data.initialize_dof_vector(invm);
    FEEvaluation<dim, degree> fe_eval(_data);
    for (unsigned int cell = 0; cell < _data.n_cell_batches(); ++cell) {
      fe_eval.reinit(cell);
      for (const unsigned int q : fe_eval.quadrature_point_indices())
        fe_eval.submit_value(make_vectorized_array(1.0), q);
      fe_eval.integrate(EvaluationFlags::values);
      fe_eval.distribute_local_to_global(invm);
    }

    invm.compress(VectorOperation::add);
    for (unsigned int k = 0; k < invm.locally_owned_size(); ++k)
      invm.local_element(k) =
          invm.local_element(k) >
                  10.0 * std::numeric_limits<RealType>::epsilon()
              ? 1.0 / invm.local_element(k)
              : 1.0;
  };

  void apply(Vector &dst, const Vector &src) const {
    _data.cell_loop(&Operator::local_apply, this, dst, src, true);
    for (unsigned int i = 0; i < dst.n_blocks(); ++i) {
      dst.block(i).scale(invm);
    }
  };

private:
  const MF &_data;
  Vector2 invm;

  void
  local_apply(const MF &data, Vector &dst, const Vector &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const {
    AssertDimension(src.n_blocks(), dst.n_blocks());
    FEEvaluation<dim, degree> fe_eval(_data);
    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src, EvaluationFlags::values);
      for (const unsigned int q : fe_eval.quadrature_point_indices()) {
        fe_eval.submit_value(fe_eval.get_value(q) + 1.0, q);
      }
      fe_eval.integrate_scatter(EvaluationFlags::values, dst);
    }
  };
};

class Problem {
public:
  using MF = MatrixFree<dim, RealType>;
  using Vector = LinearAlgebra::distributed::BlockVector<RealType>;
  using Vector2 = LinearAlgebra::distributed::Vector<RealType>;

  Problem();
  void run();

private:
  ConditionalOStream pcout;

  void make_grid_and_dofs();
  void output_results(const unsigned int increment);

  parallel::distributed::Triangulation<dim> triangulation;
  const FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;
  const MappingQ1<dim> mapping;

  AffineConstraints<RealType> constraints;

  MF data;

  Vector solution, old_solution;
};

Problem::Problem()
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef DEAL_II_WITH_P4EST
      ,
      triangulation(MPI_COMM_WORLD)
#endif
      ,
      fe(QGaussLobatto<1>(degree + 1)), dof_handler(triangulation) {
}

void Problem::make_grid_and_dofs() {
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(n_refinements);
  dof_handler.distribute_dofs(fe);

  IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
  constraints.clear();
  constraints.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  typename MF::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
      MF::AdditionalData::TasksParallelScheme::partition_partition;

  data.reinit(mapping, dof_handler, constraints, QGaussLobatto<1>(degree + 1),
              additional_data);

  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners(
      n_blocks);
  for (unsigned int i = 0; i < n_blocks; ++i) {
    partitioners[i] = data.get_vector_partitioner();
  }

  solution.reinit(partitioners);
  old_solution.reinit(partitioners);
}

void Problem::output_results(const unsigned int increment) {
  for (unsigned int i = 0; i < n_blocks; ++i) {
    constraints.distribute(solution.block(i));
  }
  solution.update_ghost_values();

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  for (unsigned int i = 0; i < n_blocks; ++i) {
    data_out.add_data_vector(solution.block(i), "n" + std::to_string(i));
  }
  data_out.build_patches(mapping);

  data_out.write_vtu_with_pvtu_record("./", "solution", increment,
                                      MPI_COMM_WORLD, 3);
  solution.zero_out_ghost_values();
}

void Problem::run() {
  make_grid_and_dofs();
  output_results(0);
  Operator op(data);
  for (unsigned int step = 1; step < n_steps; ++step) {
    pcout << "Value " << solution.l1_norm() / dof_handler.n_dofs() << std::endl;
    op.apply(solution, old_solution);
    old_solution.swap(solution);

    output_results(step);
  }
}

int main(int argc, char **argv) {
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, numbers::invalid_unsigned_int);

  try {
    Problem problem;
    problem.run();
  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
