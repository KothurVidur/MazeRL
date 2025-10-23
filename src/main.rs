mod model;

use eframe::egui;
use egui::Color32;
use std::collections::HashSet;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

#[derive(PartialEq)]
enum AppMode {
    ConfiguringMaze,
    ConfiguringTraining,
    TrainingLoop,
    DisplayOutput
}

#[derive(PartialEq)]
enum Mode {
    Rest,
    Hover,
    Start,
    Finish
}

struct MazeApp {
    rows: usize,
    cols: usize,
    temp_rows: usize,
    temp_cols: usize,
    cells: Vec<Vec<bool>>,
    hovered: Vec<Vec<bool>>,
    app_mode: AppMode,
    mode: Mode,
    start: (usize, usize),
    finish: (usize, usize),
    board: Option<model::Board>,
    error: String,
    training_num: u32,
    trajectory_limit: u32,
    discount_rate: f64,
    learning_rate: f64,
    epsilon: f64,
    trajectory: Vec<((usize, usize), (usize, usize))>,
    currently_training: bool,
    rx: Option<mpsc::Receiver<Vec<((usize, usize), (usize, usize))>>>,
    tx: Option<mpsc::Sender<Vec<((usize, usize), (usize, usize))>>>,
    progress: Arc<Mutex<f32>>
}

impl Default for MazeApp {
    fn default() -> Self {
        let rows = 10;
        let cols = 10;
        let (tx, rx) = mpsc::channel();
        Self {
            rows,
            cols,
            temp_rows: rows,
            temp_cols: cols,
            cells: vec![vec![false; cols]; rows],
            hovered: vec![vec![false; cols]; rows],
            app_mode: AppMode::ConfiguringMaze,
            mode: Mode::Rest,
            start: (0, 0),
            finish: (0, 0),
            board: None,
            error: String::from(""),
            training_num: 10000,
            trajectory_limit: 1000,
            discount_rate: 1.0,
            learning_rate: 0.1,
            epsilon: 0.9,
            trajectory: Vec::new(),
            currently_training: false,
            rx: Some(rx),
            tx: Some(tx),
            progress: Arc::new(Mutex::new(0.0))
        }
    }
}

impl eframe::App for MazeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        match self.app_mode {
            AppMode::ConfiguringMaze => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.heading("Maze Editor");

                    ui.horizontal(|ui| {
                        ui.label("Rows:");
                        ui.add(egui::DragValue::new(&mut self.temp_rows).range(2..=30));
                        ui.label("Cols:");
                        ui.add(egui::DragValue::new(&mut self.temp_cols).range(2..=30));

                        if ui.button("Resize/Reset").clicked() {
                            self.rows = self.temp_rows;
                            self.cols = self.temp_cols;
                            self.cells = vec![vec![false; self.cols]; self.rows];
                            self.hovered = vec![vec![false; self.cols]; self.rows];
                            self.start = (0, 0);
                            self.finish = (0, 0);
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.mode, Mode::Rest, "No Edit");
                        ui.radio_value(&mut self.mode, Mode::Hover, "Hover Toggle");
                        ui.radio_value(&mut self.mode, Mode::Start, "Select Start");
                        ui.radio_value(&mut self.mode, Mode::Finish, "Select Finish");
                    });

                    ui.separator();

                    egui::Grid::new("maze_grid")
                        .spacing([5.0, 4.0])
                        .show(ui, |ui| {
                            for i in 0..self.rows {
                                for j in 0..self.cols {
                                    let is_blocked = &mut self.cells[i][j];
                                    let color = if (i + 1, j + 1) == self.start {
                                        Color32::from_rgb(80, 200, 80)
                                    } else {
                                        if (i + 1, j + 1) == self.finish {
                                            Color32::from_rgb(80, 80, 200)
                                        } else {
                                            if *is_blocked {
                                                Color32::from_rgb(200, 80, 80)
                                            } else {
                                                Color32::from_rgb(255, 255, 255)
                                            }
                                        }
                                    };
                                    let button = egui::Button::new("").fill(color).min_size(egui::vec2(20.0, 20.0));
                                    let response = ui.add(button);

                                    let hover_enabled = (self.mode == Mode::Hover)
                                        && ((i + 1, j + 1) != self.start)
                                        && ((i + 1, j + 1) != self.finish);

                                    if response.hovered() {
                                        if hover_enabled {
                                            if !self.hovered[i][j] {
                                                *is_blocked = !*is_blocked;
                                                self.hovered[i][j] = true;
                                            }
                                        }
                                    } else {
                                        self.hovered[i][j] = false;
                                    }

                                    if response.clicked() {
                                        match self.mode {
                                            Mode::Start => {
                                                if (i + 1, j + 1) == self.start {
                                                    self.start = (0, 0);
                                                } else {
                                                    if (i + 1, j + 1) != self.finish {
                                                        self.start = (i + 1, j + 1);
                                                    }
                                                }
                                                self.cells[i][j] = false;
                                            },
                                            Mode::Finish => {
                                                if (i + 1, j + 1) == self.finish {
                                                    self.finish = (0, 0);
                                                } else {
                                                    if (i + 1, j + 1) != self.start {
                                                        self.finish = (i + 1, j + 1);
                                                    }
                                                }
                                                self.cells[i][j] = false;
                                            },
                                            _ => ()
                                        }
                                    }
                                }
                                ui.end_row();
                            }
                        });

                    ui.separator();

                    if ui.button("Proceed to Training").clicked() {
                        if self.start == (0, 0) {
                            self.error = String::from("Please select a starting point before proceeding.");
                        } else if self.finish == (0, 0) {
                            self.error = String::from("Please select a finishing point before proceeding.");
                        } else {
                            self.error = String::from("");
                            let mut set: HashSet<(usize, usize)> = HashSet::new();
                            for i in 0..self.rows {
                                for j in 0..self.cols {
                                    if self.cells[i][j] {
                                        set.insert((i + 1, j + 1));
                                    }
                                }
                            }
                            self.board = Some(model::Board::new(self.rows, self.cols, self.start, self.finish, &set));
                            self.app_mode = AppMode::ConfiguringTraining;
                        }
                    }

                    ui.label(format!("{}", self.error));
                });
            },
            AppMode::ConfiguringTraining => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.heading("Maze Editor");

                    ui.horizontal(|ui| {
                        ui.label("Training Steps:");
                        ui.add(egui::DragValue::new(&mut self.training_num).speed(1000).range(1..=100000));
                        ui.label("Trajectory Limit:");
                        ui.add(egui::DragValue::new(&mut self.trajectory_limit).speed(10).range(1..=1000));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Discount Rate:");
                        ui.add(egui::DragValue::new(&mut self.discount_rate).speed(0.01).range(0.0..=1.0));
                        ui.label("Learning Rate:");
                        ui.add(egui::DragValue::new(&mut self.learning_rate).speed(0.01).range(0.0..=1.0));
                        ui.label("Epsilon:");
                        ui.add(egui::DragValue::new(&mut self.epsilon).speed(0.01).range(0.0..=1.0));
                    });

                    ui.separator();

                    egui::Grid::new("maze_grid")
                        .spacing([5.0, 4.0])
                        .show(ui, |ui| {
                            for i in 0..self.rows {
                                for j in 0..self.cols {
                                    let is_blocked = &mut self.cells[i][j];
                                    let color = if (i + 1, j + 1) == self.start {
                                        Color32::from_rgb(80, 200, 80)
                                    } else {
                                        if (i + 1, j + 1) == self.finish {
                                            Color32::from_rgb(80, 80, 200)
                                        } else {
                                            if *is_blocked {
                                                Color32::from_rgb(200, 80, 80)
                                            } else {
                                                Color32::from_rgb(255, 255, 255)
                                            }
                                        }
                                    };
                                    ui.add(egui::Button::new("").fill(color).min_size(egui::vec2(20.0, 20.0)));
                                }
                                ui.end_row();
                            }
                        });

                    ui.separator();

                    if ui.button("Begin Training Loop").clicked() {
                        self.app_mode = AppMode::TrainingLoop;
                    }
                });
            },
            AppMode::TrainingLoop => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.heading("Maze Editor");

                    ui.separator();

                    ui.label("Training...");

                    let progress_value = *self.progress.lock().unwrap();

                    ui.add(
                        egui::ProgressBar::new(progress_value)
                            .show_percentage()
                            .text(format!("{:.1}%", progress_value * 100.0))
                    );
                });
                
                let training_num_c = self.training_num;
                let trajectory_limit_c = self.trajectory_limit;
                let discount_rate_c = self.discount_rate;
                let learning_rate_c = self.learning_rate;
                let epsilon_c = self.epsilon;

                if !self.currently_training {
                    self.currently_training = true;

                    let board = self.board.clone();
                    let tx = self.tx.clone().unwrap(); 
                    let progress = Arc::clone(&self.progress);

                    thread::spawn(move || {
                        if let Some(mut b) = board {
                            for i in 0..training_num_c {
                                b.train(1, trajectory_limit_c, discount_rate_c, learning_rate_c, epsilon_c);
                                {
                                    let mut p = progress.lock().unwrap();
                                    *p = (i as f32 + 1.0) / training_num_c as f32;
                                }
                            }
                            let trajectory = b.trajectory(1000, 0.0);
                            tx.send(trajectory).unwrap();
                        }
                    });
                }

                if let Some(rx) = &self.rx {
                    if let Ok(result) = rx.try_recv() {
                        self.trajectory = result;
                        self.app_mode = AppMode::DisplayOutput;
                        self.currently_training = false;
                    }
                }

                ctx.request_repaint();
            },
            AppMode::DisplayOutput => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let cell_size_x = 44.0;
                    let cell_size_y = 24.0;
                    let grid_rect = ui.min_rect();

                    ui.heading("Maze Editor");

                    ui.separator();

                    egui::Grid::new("maze_grid")
                        .spacing([4.0, 4.0])
                        .show(ui, |ui| {
                            for i in 0..self.rows {
                                for j in 0..self.cols {
                                    let is_blocked = &mut self.cells[i][j];
                                    let color = if (i + 1, j + 1) == self.start {
                                        Color32::from_rgb(80, 200, 80)
                                    } else {
                                        if (i + 1, j + 1) == self.finish {
                                            Color32::from_rgb(80, 80, 200)
                                        } else {
                                            if *is_blocked {
                                                Color32::from_rgb(200, 80, 80)
                                            } else {
                                                Color32::from_rgb(255, 255, 255)
                                            }
                                        }
                                    };
                                    ui.add(egui::Button::new("").fill(color).min_size(egui::vec2(20.0, 20.0)));
                                }
                                ui.end_row();
                            }
                        });

                    let painter = ui.painter();

                    let offset = grid_rect.min.to_vec2();

                    for (from, to) in self.trajectory.iter() {
                        let from_pos = egui::pos2(
                            offset.x + from.1 as f32 * cell_size_x + 10.0,
                            offset.y + from.0 as f32 * cell_size_y + 42.0,
                        );

                        let to_pos = egui::pos2(
                            offset.x + to.1 as f32 * cell_size_x + 11.0,
                            offset.y + to.0 as f32 * cell_size_y + 42.0,
                        );

                        painter.line_segment([from_pos, to_pos], egui::Stroke::new(2.0, egui::Color32::ORANGE));
                    }
                });
            }
        };
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "RL Maze",
        options,
        Box::new(|_cc| Ok(Box::<MazeApp>::default())),
    )
}
