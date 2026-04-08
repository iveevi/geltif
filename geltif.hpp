#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <filesystem>
#include <unordered_map>
#include <variant>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>


struct Geometry {
	std::vector <glm::vec3> positions;
	std::vector <glm::vec3> normals;
	std::vector <glm::vec2> uvs;
	std::vector <glm::uvec3> triangles;

	void recalculate_normals();
	void deduplicate();
};

struct Material {
	std::string albedo;
	std::string metallic_roughness;
	std::string normals;
	std::string occlusion;
	std::string emissive;

	glm::vec3 base_color { 1.0f };
	glm::vec3 emissive_factor { 0.0f };
	float emissive_strength = 1.0f;
	float metallic_factor = 1.0f;
	float roughness_factor = 1.0f;
	float alpha_cutoff = 0.0f;
};

struct DirectionalLight {
	glm::vec3 color { 1.0f };
	float intensity = 1.0f;
	glm::vec3 direction { 0, 0, -1 };
};

struct PointLight {
	glm::vec3 color { 1.0f };
	float intensity = 1.0f;
	glm::vec3 position { 0.0f };
	float range = 0.0f;
};

struct SpotLight {
	glm::vec3 color { 1.0f };
	float intensity = 1.0f;
	glm::vec3 position { 0.0f };
	glm::vec3 direction { 0, 0, -1 };
	float range = 0.0f;
	float inner_cone = 0.0f;
	float outer_cone = 0.7854f;
};

using Light = std::variant <DirectionalLight, PointLight, SpotLight>;


struct AnimationChannel {
	std::vector <float> times;
	std::vector <glm::vec4> values;
};

struct Animation {
	AnimationChannel translation;
	AnimationChannel rotation;
	AnimationChannel scale;
	float duration = 0.0f;
	float current_time = 0.0f;

	// Node's local rest-pose TRS (fallback for unanimated channels)
	glm::vec3 rest_translation { 0.0f };
	glm::quat rest_rotation { 1.0f, 0.0f, 0.0f, 0.0f };
	glm::vec3 rest_scale { 1.0f };

	void advance(float dt);
	glm::mat4 evaluate() const;
};

struct Camera {
	enum class Type {
		ePerspective,
		eOrthographic
	} type = Type::ePerspective;

	float fov = glm::radians(45.0f);
	float aspect = 1.0f;
	float xmag = 1.0f;
	float ymag = 1.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	glm::mat4 transform { 1.0f };
	glm::mat4 parent_transform { 1.0f };
	std::optional <Animation> animation;

	struct RayFrame {
		glm::vec3 u;
		glm::vec3 v;
		glm::vec3 w;
		glm::vec3 origin;
	};

	void update(float dt);
	glm::mat4 view_matrix() const;
	glm::mat4 projection_matrix(float aspect_override) const;
	glm::vec3 eye_position() const;
	RayFrame ray_frame(float aspect_override) const;
};

struct Entity {
	uint32_t geometry_index;
	uint32_t material_index;
	glm::mat4 transform { 1.0f };
};

struct RawTexture {
	uint32_t width;
	uint32_t height;
	std::vector <uint8_t> pixels;
};

struct Scene {
	std::vector <Geometry> geometries;
	std::vector <Material> materials;
	std::vector <Entity> entities;
	std::vector <Light> lights;
	std::vector <Camera> cameras;
	std::unordered_map <std::string, RawTexture> textures;

	// TODO: should get rid of this merge method soon...
	Geometry merge() const;

	// TODO: probably wont need this method either
	std::pair <glm::vec3, glm::vec3> bounds() const;

	static Scene from_file(const std::filesystem::path &path);
};
